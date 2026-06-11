import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import *


class DeformableCAB(nn.Module):
    def __init__(self, dim, num_heads, bias, offset_groups=4):
        super(DeformableCAB, self).__init__()
        self.num_heads = num_heads
        self.offset_groups = offset_groups
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Query projection
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Key/Value projection with deformable offsets
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # Offset prediction
        self.offset_conv = nn.Sequential(
            nn.Conv2d(dim, offset_groups * 2, kernel_size=3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(offset_groups * 2, offset_groups * 2, kernel_size=3, padding=1, bias=bias)
        )

        if self.offset_conv[-1].weight is not None:
            nn.init.constant_(self.offset_conv[-1].weight, 0)
        if self.offset_conv[-1].bias is not None:
            nn.init.constant_(self.offset_conv[-1].bias, 0)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # Query from input x
        q = self.q_dwconv(self.q(x))

        # Predict deformable offsets from context y
        offset = self.offset_conv(y)  # [b, offset_groups*2, h, w]
        offset = rearrange(offset, 'b (g c) h w -> b g c h w', g=self.offset_groups, c=2)

        # Deformable key/value sampling
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        # Apply deformable offsets
        k = self.apply_deformable_offset(k, offset)
        v = self.apply_deformable_offset(v, offset)

        # Multi-head attention
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

    def apply_deformable_offset(self, feature, offset):
        """Apply deformable offsets to feature map"""
        b, c, h, w = feature.shape
        device = feature.device

        # Generate base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)  # [h, w, 2]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [b, h, w, 2]

        # Apply offsets (group-wise)
        feature = rearrange(feature, 'b (g c) h w -> b g c h w', g=self.offset_groups)
        deformed = []
        for gi in range(self.offset_groups):
            offset_group = offset[:, gi]  # [b, 2, h, w]
            offset_group = offset_group.permute(0, 2, 3, 1)  # [b, h, w, 2]

            # Normalize offsets to [-1,1] range
            offset_group[:, :, :, 0] = offset_group[:, :, :, 0] / (w - 1) * 2
            offset_group[:, :, :, 1] = offset_group[:, :, :, 1] / (h - 1) * 2

            deformed_grid = grid + offset_group
            deformed_feat = F.grid_sample(
                feature[:, gi],
                deformed_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )
            deformed.append(deformed_feat)

        return torch.cat(deformed, dim=1)


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = DeformableCAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x)) + x
        return x


class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = DeformableCAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x