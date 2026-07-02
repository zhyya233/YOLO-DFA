
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import CondConv2d

__all__ = ["Conv", "DynamicConv", "Bottleneck_DynamicConv", "C2f_DynamicConv"]


DEBUG = os.environ.get("YF_DEBUG_TENSORS", "0") == "1"


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def _check_finite(x, msg="Tensor contains non-finite values"):
    if DEBUG and (torch.isnan(x).any() or torch.isinf(x).any()):
        raise RuntimeError(msg)


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        _check_finite(out, "Conv output contains non-finite values")
        return out

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DynamicConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(c1, num_experts)

        pad = autopad(k, p, d)
        if isinstance(pad, (list, tuple)):
            pad = int(pad[0])

        self.cond_conv = CondConv2d(
            c1,
            c2,
            k,
            s,
            pad,
            dilation=d,
            groups=g,
            bias=False,
            num_experts=num_experts,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.eps = 1e-8

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.routing(pooled_inputs)

        logits = logits - logits.max(dim=1, keepdim=True)[0]
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        routing_weights = F.softmax(logits, dim=1)
        routing_weights = torch.clamp(routing_weights, min=self.eps, max=1.0)
        routing_weights = routing_weights.to(dtype=x.dtype, device=x.device)

        if DEBUG and not torch.isfinite(routing_weights).all():
            raise RuntimeError("DynamicConv routing weights contain non-finite values")

        out = self.act(self.bn(self.cond_conv(x, routing_weights)))
        _check_finite(out, "DynamicConv output contains non-finite values")
        return out

    def forward_fuse(self, x):
        return self.forward(x)


class Bottleneck_DynamicConv(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = DynamicConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        branch = self.cv2(self.cv1(x))
        if self.add:
            _check_finite(branch, "Bottleneck_DynamicConv branch contains non-finite values")
            return x + branch
        return branch


class C2f_DynamicConv(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_DynamicConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split(self.c, 1))
        for module in self.m:
            y.append(module(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        _check_finite(out, "C2f_DynamicConv output contains non-finite values")
        return out
