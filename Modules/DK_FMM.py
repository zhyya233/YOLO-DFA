import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
import os

__all__ = ['DK_FMM']


DEBUG = os.environ.get('YF_DEBUG_TENSORS', '0') == '1'


def _check_finite(x, msg="Tensor contains non-finite values"):
    if DEBUG and (torch.isnan(x).any() or torch.isinf(x).any()):
        raise RuntimeError(msg)


class DK_FMM(nn.Module):

    def __init__(self,
                 dim: int,
                 focal_level: int = 2,
                 focal_window: int = 3,
                 focal_factor: int = 2,
                 proj_drop: float = 0.):
        super().__init__()

        self.dim = dim
        self.focal_level = focal_level



        self.projection = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1)
        self.context_transform = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)


        self.focal_layers = nn.ModuleList()
        self.offset_predictors = nn.ModuleList()

        for k in range(self.focal_level):
            kernel_size = focal_factor * k + focal_window



            self.focal_layers.append(
                DeformConv2d(dim, dim, kernel_size=kernel_size, stride=1,
                             padding=kernel_size // 2, groups=dim, bias=False)
            )



            out_channels_offset = 2 * kernel_size * kernel_size


            groups = 4
            if dim % groups != 0 or out_channels_offset % groups != 0:
                groups = 1

            offset_conv = nn.Conv2d(dim, out_channels_offset, kernel_size=3, stride=1, padding=1, groups=groups)



            nn.init.constant_(offset_conv.weight, 0.0)
            nn.init.constant_(offset_conv.bias, 0.0)

            self.offset_predictors.append(offset_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.numel() == 0 or x.shape[2] * x.shape[3] == 0:
            return x


        x_proj = self.projection(x)
        q, ctx, gates = torch.split(x_proj, (self.dim, self.dim, self.focal_level + 1), 1)


        gates = torch.sigmoid(gates)


        ctx_all = 0.0


        offsets = self.offset_predictors[0](ctx)

        for l in range(self.focal_level):

            if DEBUG: _check_finite(offsets, f"Offset predictor level {l} produced NaN")



            ctx = self.focal_layers[l](ctx, offsets)
            ctx = self.act(ctx)



            if l < self.focal_level - 1:
                offsets = self.offset_predictors[l + 1](ctx)



            ctx_all = ctx_all + ctx * gates[:, l:l + 1]



        ctx_global = self.act(ctx.mean(dim=[2, 3], keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]


        modulator = self.context_transform(ctx_all)


        x_out = q * modulator


        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)

        if DEBUG: _check_finite(x_out, "DK_FMM Output NaN")

        return x_out
