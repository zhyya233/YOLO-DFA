

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.layers import SqueezeExcite, DropPath


DEBUG = os.environ.get('YF_DEBUG_TENSORS', '0') == '1'
inplace = True

def _check_finite(x, msg="Tensor contains non-finite values"):
    if DEBUG and (torch.isnan(x).any() or torch.isinf(x).any()):
        raise RuntimeError(msg)

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x

def get_norm(norm_layer='in_1d'):
    eps = 1e-6

    def identity_factory(dim): return nn.Identity()
    norm_dict = {
        'none': identity_factory,
        'in_1d': lambda dim: nn.InstanceNorm1d(dim, eps=eps),
        'in_2d': lambda dim: nn.InstanceNorm2d(dim, eps=eps),
        'in_3d': lambda dim: nn.InstanceNorm3d(dim, eps=eps),
        'bn_1d': lambda dim: nn.BatchNorm1d(dim, eps=eps),
        'bn_2d': lambda dim: nn.BatchNorm2d(dim, eps=eps),
        'bn_3d': lambda dim: nn.BatchNorm3d(dim, eps=eps),
        'gn': lambda dim: nn.GroupNorm(32 if dim >= 32 else 1, dim, eps=eps),
        'ln_1d': lambda dim: nn.LayerNorm(dim, eps=eps),
        'ln_2d': lambda dim: LayerNorm2d(dim, eps=eps),
    }
    if norm_layer not in norm_dict:
        raise KeyError(f"Unknown norm_layer '{norm_layer}'")
    return norm_dict[norm_layer]

def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'silu': nn.SiLU
    }
    if act_layer not in act_dict:
        raise KeyError(f"Unknown act_layer '{act_layer}'")
    return act_dict[act_layer]

class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        act_cls = get_act(act_layer)

        try:
            self.act = act_cls(inplace=inplace)
        except Exception:
            try:
                self.act = act_cls()
            except Exception:
                self.act = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        _check_finite(x, "ConvNormAct produced non-finite values")
        return x

class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
                 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0,
                 dim_head=8, window_size=7, attn_s=True, qkv_bias=False, attn_drop=0., drop=0.,
                 drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        self.eps = 1e-6
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
            self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias,
                                 norm_layer='none', act_layer=act_layer, inplace=inplace)
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
            else:
                self.v = nn.Identity()

        self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation,
                                      groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
        self.se = SqueezeExcite(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)

        self.proj_bn = get_norm('bn_2d')(dim_out)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def _stable_softmax(self, x, dim=-1):
        x = x - x.max(dim=dim, keepdim=True)[0]
        x = torch.clamp(x, min=-50.0, max=50.0)
        x_exp = torch.exp(x)
        denom = x_exp.sum(dim=dim, keepdim=True) + self.eps
        return x_exp / denom

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape

        if self.attn_s:
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x_pad = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x_blocks = rearrange(x_pad, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()

            b_blk, c_blk, h_blk, w_blk = x_blocks.shape
            qk = self.qk(x_blocks)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head',
                           qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]

            q = torch.clamp(q, -15.0, 15.0)
            k = torch.clamp(k, -15.0, 15.0)

            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = self._stable_softmax(attn_spa, dim=-1)
            attn_spa = self.attn_drop(attn_spa)

            if self.attn_pre:
                x_raw = rearrange(x_blocks, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x_raw
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h_blk, w=w_blk).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x_blocks)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h_blk, w=w_blk).contiguous()

            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        local = self.conv_local(x)
        _check_finite(local, "iRMB conv_local produced non-finite values")
        if self.has_skip:
            x = x + self.se(local)
        else:
            x = self.se(local)

        x = self.proj_drop(x)
        x = self.proj(x)
        x = self.proj_bn(x)
        _check_finite(x, "iRMB proj produced non-finite values")

        if self.has_skip:
            if DEBUG and (torch.isnan(shortcut).any() or torch.isinf(shortcut).any()):
                raise RuntimeError("iRMB shortcut contains non-finite values")
            if DEBUG and (torch.isnan(x).any() or torch.isinf(x).any()):
                raise RuntimeError("iRMB body contains non-finite values before residual add")
            x = shortcut + self.drop_path(x)
        return x

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
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        _check_finite(out, "Conv produced non-finite values")
        return out

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.iRMB = iRMB(c2, c2)

    def forward(self, x):
        out = self.iRMB(self.cv2(self.cv1(x)))
        if self.add:
            if DEBUG and (torch.isnan(out).any() or torch.isinf(out).any()):
                raise RuntimeError("Bottleneck: branch produced non-finite values before residual add")
            return x + out
        else:
            return out

class C2f_iRMB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        _check_finite(out, "C2f_iRMB produced non-finite values")
        return out

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        for m in self.m:
            y.append(m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        _check_finite(out, "C2f_iRMB produced non-finite values (split)")
        return out

