import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.layers import DropPath, SqueezeExcite

__all__ = ['iRMB_EMA', 'C2f_iRMB_EMA', 'C2f_Bifocal']





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
        assert self.channels_per_group > 0


        self.softmax = nn.Softmax(-1)

        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(self.channels_per_group, self.channels_per_group)
        self.conv1x1 = nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()


        if h * w == 0:
            return x

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





inplace = True


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):

        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def get_norm(norm_layer='in_1d'):
    eps = 1e-6

    norm_dict = {'none': nn.Identity, 'bn_2d': partial(nn.BatchNorm2d, eps=eps), 'gn': partial(nn.GroupNorm, eps=eps)}
    return norm_dict.get(norm_layer, nn.Identity)(dim_out) if isinstance(norm_dict.get(norm_layer),
                                                                         partial) else norm_dict.get(norm_layer,
                                                                                                     nn.Identity)


def get_act(act_layer='relu'):
    act_dict = {'none': nn.Identity, 'relu': nn.ReLU, 'silu': nn.SiLU}
    return act_dict.get(act_layer, nn.ReLU)


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, skip=False,
                 norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super().__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)

        self.norm = nn.BatchNorm2d(dim_out) if norm_layer == 'bn_2d' else nn.Identity()
        self.act = nn.SiLU(inplace=inplace) if act_layer == 'silu' else nn.ReLU(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class iRMB_EMA(nn.Module):
    def __init__(self, dim_in, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d', act_layer='relu',
                 v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, attn_s=True, qkv_bias=False, drop=0.,
                 drop_path=0.):
        super().__init__()
        dim_out = dim_in
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s

        if self.attn_s:
            self.ema = EMA(dim_in)
        else:
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none',
                                 act_layer=act_layer) if v_proj else nn.Identity()

        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu')
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio) if se_ratio > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none')
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x_norm = self.norm(x)
        v = self.ema(x_norm) if self.attn_s else self.v(x_norm)
        x_res = self.se(self.conv_local(v))
        x = v + x_res
        x = self.proj(self.proj_drop(x))
        return (shortcut + self.drop_path(x)) if self.has_skip else x



def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, d=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, d=d)
        self.add = shortcut and c1 == c2
        self.iRMB = iRMB_EMA(c2, dilation=d)

    def forward(self, x):
        return x + self.iRMB(self.cv2(self.cv1(x))) if self.add else self.iRMB(self.cv2(self.cv1(x)))


class C2f_iRMB_EMA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))





class C2f_Bifocal(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, d=2):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.m_details = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0, d=1) for _ in range(n))
        self.m_context = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0, d=d) for _ in range(n))

        self.gate = nn.Sequential(
            nn.Linear(2 * self.c, self.c // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.c // 4, 2 * self.c),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.cv2 = Conv(2 * self.c, c2, 1)

    def forward(self, x):

        if x.size(2) * x.size(3) == 0:
            return self.cv2(self.cv1(x))

        y_split = self.cv1(x)
        y_base, y_proc = y_split.chunk(2, 1)

        x_details = y_proc
        for m in self.m_details:
            x_details = m(x_details)

        x_context = y_proc
        for m in self.m_context:
            x_context = m(x_context)


        pooled_features = torch.cat((
            F.adaptive_avg_pool2d(x_details, 1),
            F.adaptive_avg_pool2d(x_context, 1)
        ), dim=1).view(x.size(0), -1)

        attention_weights = self.gate(pooled_features).view(x.size(0), 2 * self.c, 1, 1)


        attention_weights = torch.clamp(attention_weights, min=1e-6, max=1.0)

        w_details, w_context = attention_weights.chunk(2, 1)
        fused_features = x_details * w_details + x_context * w_context

        y_out = torch.cat((y_base, y_base + self.gamma * fused_features), 1)

        return self.cv2(y_out)