import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Enhanced_Parallel_Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm2 = nn.BatchNorm2d(dim)

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )


        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class GuidedMultiScaleFrequencyAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.low_freq = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.high_freq = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.low_guidance = CrossChannelAttention(channels // 8, channels // 2)
        self.high_guidance = CrossChannelAttention(channels // 8, channels // 2)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels // 8 + channels // 2 + channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        low = self.low_freq(x)
        mid = self.mid_freq(x)
        high = self.high_freq(x)

        low_guided = self.low_guidance(low, mid)
        high_guided = self.high_guidance(high, mid)

        fused = torch.cat([low_guided, mid, high_guided], dim=1)
        return self.fusion(fused)


class CrossChannelAttention(nn.Module):

    def __init__(self, target_channels, guide_channels):
        super().__init__()
        self.guide_transform = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(guide_channels, target_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, target, guide):
        guidance_weights = self.guide_transform(guide)
        return target * (1 + guidance_weights)


class DynamicChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(4, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ResidualBlock(nn.Module):

    def __init__(self, main):
        super().__init__()
        self.main = main

    def forward(self, x):
        return x + self.main(x)


class MGMFE(nn.Module):
    def __init__(self, nc, expand=2):
        super().__init__()
        self.nc = nc

        self.freq_filter = nn.Parameter(torch.ones(1, nc, 1, 1))

        self.process_stages = Enhanced_Parallel_Attention(nc)

        self.gate = nn.Sequential(
            GuidedMultiScaleFrequencyAttention(nc),
            DynamicChannelAttention(nc)
        )

        self.register_buffer('scharr_x', torch.tensor([[-3, 0, 3],
                                                       [-10, 0, 10],
                                                       [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('scharr_y', torch.tensor([[-3, -10, -3],
                                                       [0, 0, 0],
                                                       [3, 10, 3]], dtype=torch.float32).view(1, 1, 3, 3))

        self.edge_alpha = nn.Parameter(torch.zeros(1, nc, 1, 1))

        self.phase_gamma = nn.Parameter(torch.tensor(1.0))

        self.freq_residual = nn.Conv2d(nc, nc, 1, bias=False)

    def compute_gradient(self, x):

        grad_x = F.conv2d(x, self.scharr_x.repeat(self.nc, 1, 1, 1),
                          padding=1, groups=self.nc)
        grad_y = F.conv2d(x, self.scharr_y.repeat(self.nc, 1, 1, 1),
                          padding=1, groups=self.nc)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        mag = mag * self.freq_filter

        mag_processed = self.process_stages(mag) + self.freq_residual(mag)

        gate_weight = self.gate(mag_processed)
        mag = mag_processed * (1 + gate_weight)

        edge_grad = self.compute_gradient(x)
        edge_freq = torch.fft.rfft2(edge_grad, norm='backward')
        edge_mag = torch.abs(edge_freq)
        mag = mag + self.edge_alpha * edge_mag

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return identity * self.phase_gamma + x_out


class Frequency_Domain(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.freq = MGMFE(channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 16), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 16), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        att = self.channel_att(x)
        x = x * att
        x = self.norm(x)
        x = self.freq(x)
        return identity + self.gamma * x


