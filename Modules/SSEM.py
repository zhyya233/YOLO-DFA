import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SSEM', 'SCAM']

DEBUG = os.environ.get('YF_DEBUG_TENSORS', '0') == '1'


def _check_finite(x, msg="Tensor contains non-finite values"):
    if DEBUG and (torch.isnan(x).any() or torch.isinf(x).any()):
        raise RuntimeError(msg)


class SSEM(nn.Module):

    def __init__(self, c, use_bn: bool = True, clamp_val: float = 15.0):
        super().__init__()
        self.conv_qk = nn.Conv2d(c, 1, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.conv_combiner = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.use_bn = use_bn
        self.bn = nn.BatchNorm2d(c) if use_bn else nn.Identity()
        self.eps = 1e-6
        self.clamp_val = float(clamp_val)

    def _stable_softmax(self, x, dim: int):

        x = x - x.max(dim=dim, keepdim=True)[0]
        x = torch.clamp(x, min=-50.0, max=50.0)
        x_exp = torch.exp(x)
        denom = x_exp.sum(dim=dim, keepdim=True).clamp_min(self.eps)
        return x_exp / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()



        qk = self.conv_qk(x).view(b, 1, h * w)
        qk = torch.clamp(qk, -self.clamp_val, self.clamp_val)
        spatial_attn_weights = self._stable_softmax(qk, dim=2)
        _check_finite(spatial_attn_weights, "SSEM spatial_attn_weights non-finite")


        v = self.conv_v(x).view(b, c, h * w)
        spatial_out = torch.bmm(v, spatial_attn_weights.transpose(1, 2)).view(b, c, 1, 1)
        spatial_out = spatial_out.expand(-1, -1, h, w)


        gap = F.adaptive_avg_pool2d(x, 1)
        gmp = F.adaptive_max_pool2d(x, 1)

        channel_feats = torch.cat([gap, gmp], dim=2).view(b, c, 2)
        channel_feats = torch.clamp(channel_feats, -self.clamp_val, self.clamp_val)
        channel_attn_weights = self._stable_softmax(channel_feats, dim=2)
        _check_finite(channel_attn_weights, "SSEM channel_attn_weights non-finite")

        w_avg = channel_attn_weights[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        w_max = channel_attn_weights[:, :, 1].unsqueeze(-1).unsqueeze(-1)
        channel_out = gap * w_avg + gmp * w_max
        channel_out = channel_out.expand(-1, -1, h, w)


        combined = spatial_out + channel_out
        _check_finite(combined, "SSEM combined features non-finite before conv")

        combined = self.conv_combiner(combined)
        if self.use_bn:
            combined = self.bn(combined)
        _check_finite(combined, "SSEM combined features non-finite after conv/BN")

        final_attn_weights = self.sigmoid(combined)
        final_attn_weights = torch.clamp(final_attn_weights, min=self.eps, max=1.0 - self.eps)

        return x * final_attn_weights




SCAM = SSEM
