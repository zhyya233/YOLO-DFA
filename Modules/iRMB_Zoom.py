import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.layers import DropPath, SqueezeExcite
import os




__all__ = ['iRMB_Zoom', 'C2f_iRMB_Zoom', 'iRMB_EMA', 'C2f_iRMB_EMA', 'Conv', 'Bottleneck']


DEBUG = os.environ.get('YF_DEBUG_TENSORS', '0') == '1'


def _check_finite(x, msg="Tensor contains non-finite values"):
    if DEBUG and (torch.isnan(x).any() or torch.isinf(x).any()):
        raise RuntimeError(f"{msg}: {x.shape}")


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'ln_2d': partial(nn.LayerNorm, eps=eps),
    }
    return norm_dict.get(norm_layer, nn.Identity)


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'gelu': nn.GELU
    }
    return act_dict.get(act_layer, nn.ReLU)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p






class EMA(nn.Module):

    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        if factor <= 0: factor = 1
        max_candidate = min(factor, channels)
        groups = 1
        for g in range(max_candidate, 0, -1):
            if channels % g == 0:
                groups = g
                break
        self.groups = groups
        self.channels_per_group = channels // self.groups

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(self.channels_per_group, self.channels_per_group)
        self.conv1x1 = nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size=3, padding=1)

    def forward(self, x):

        if x.numel() == 0: return x

        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)


        x1_pooled = self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        x1_pooled = torch.clamp(x1_pooled, min=-20, max=20)
        x11 = self.softmax(x1_pooled)

        x12 = x2.reshape(b * self.groups, self.channels_per_group, -1)

        x2_pooled = self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1)
        x2_pooled = torch.clamp(x2_pooled, min=-20, max=20)
        x21 = self.softmax(x2_pooled)

        x22 = x1.reshape(b * self.groups, self.channels_per_group, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))






class iRMB_Zoom(nn.Module):

    def __init__(self, dim_in, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='silu', dw_ks=3, stride=1, dilation=1, se_ratio=0.0,
                 drop=0., drop_path=0.,

                 zoom_factor=1.5, hires_split_ratio=0.5):
        super().__init__()

        dim_out = dim_in
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip


        self.c_hires = int(dim_mid * hires_split_ratio)
        self.c_ctx = dim_mid - self.c_hires


        self.conv_expand = nn.Sequential(
            nn.Conv2d(dim_in, dim_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_mid),
            get_act(act_layer)(inplace=True)
        )


        self.ema = EMA(self.c_ctx)
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(self.c_ctx, self.c_ctx, kernel_size=dw_ks, stride=stride, padding=dilation,
                      dilation=dilation, groups=self.c_ctx, bias=False),
            nn.BatchNorm2d(self.c_ctx),
            get_act(act_layer)(inplace=True)
        )


        self.zoom_factor = zoom_factor
        self.region_selector = nn.Sequential(
            nn.Conv2d(self.c_hires, self.c_hires, 3, padding=1, groups=self.c_hires, bias=False),
            nn.BatchNorm2d(self.c_hires),
            nn.Sigmoid()
        )
        self.hires_refine = nn.Sequential(
            nn.Conv2d(self.c_hires, self.c_hires, 3, padding=1, groups=self.c_hires, bias=False),
            nn.BatchNorm2d(self.c_hires),
            get_act(act_layer)(inplace=True)
        )


        self.fusion_se = SqueezeExcite(dim_mid, rd_ratio=se_ratio if se_ratio > 0 else 0.25,
                                       act_layer=get_act(act_layer))


        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1, stride=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):

        if x.numel() == 0 or x.shape[2] * x.shape[3] == 0:
            return x

        shortcut = x
        x = self.norm(x)

        x_expanded = self.conv_expand(x)
        x_ctx, x_hires = torch.split(x_expanded, [self.c_ctx, self.c_hires], dim=1)


        y_ctx = self.ctx_conv(self.ema(x_ctx))


        region_map = self.region_selector(x_hires)


        h, w = x_hires.shape[2:]
        zoom_h = max(1, int(h * self.zoom_factor))
        zoom_w = max(1, int(w * self.zoom_factor))
        zoomed_size = (zoom_h, zoom_w)


        x_zoomed = F.interpolate(x_hires, size=zoomed_size, mode='bilinear', align_corners=False)
        region_map_zoomed = F.interpolate(region_map, size=zoomed_size, mode='bilinear', align_corners=False)


        region_map_zoomed = torch.clamp(region_map_zoomed, 0.0, 1.0)


        y_zoomed_refined = self.hires_refine(x_zoomed * region_map_zoomed)


        y_hires = F.interpolate(y_zoomed_refined, size=(h, w), mode='bilinear', align_corners=False)


        y_fused = torch.cat([y_ctx, y_hires], dim=1)
        y_fused_attended = self.fusion_se(y_fused)


        x = self.proj_drop(y_fused_attended)
        x = self.proj(x)

        if DEBUG: _check_finite(x, "iRMB-Zoom output NaN")

        return (shortcut + self.drop_path(x)) if self.has_skip else x


class Bottleneck_iRMB_Zoom(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.iRMB = iRMB_Zoom(c2, exp_ratio=1.0)

    def forward(self, x):
        return x + self.iRMB(self.cv2(self.cv1(x))) if self.add else self.iRMB(self.cv2(self.cv1(x)))


class C2f_iRMB_Zoom(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_iRMB_Zoom(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))






class iRMB_EMA(nn.Module):
    def __init__(self, dim_in, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d', act_layer='relu',
                 v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, attn_s=True, qkv_bias=False, drop=0.,
                 drop_path=0.):
        super().__init__()
        dim_out = dim_in
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s

        if self.attn_s:
            self.ema = EMA(dim_in)
        else:
            self.v = nn.Sequential(nn.Conv2d(dim_in, dim_mid, 1, bias=qkv_bias),
                                   get_act(act_layer)(inplace=True)) if v_proj else nn.Identity()

        self.conv_local = nn.Sequential(
            nn.Conv2d(dim_mid, dim_mid, dw_ks, stride, (dw_ks - 1) // 2, dilation, dim_mid, bias=False),
            nn.BatchNorm2d(dim_mid),
            get_act('silu')(inplace=True)
        )
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Sequential(nn.Conv2d(dim_mid, dim_out, 1, bias=False), nn.Identity())
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        if x.numel() == 0: return x
        shortcut = x
        x = self.norm(x)
        v = self.ema(x) if self.attn_s else self.v(x)
        x = v + self.se(self.conv_local(v))
        x = self.proj(self.proj_drop(x))
        return (shortcut + self.drop_path(x)) if self.has_skip else x


class Bottleneck_Legacy(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.iRMB = iRMB_EMA(c2)

    def forward(self, x):
        return x + self.iRMB(self.cv2(self.cv1(x))) if self.add else self.iRMB(self.cv2(self.cv1(x)))


class C2f_iRMB_EMA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_Legacy(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


