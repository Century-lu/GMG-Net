import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNormFunction(torch.autograd.Function):
    """高性能自定义LayerNorm实现"""

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
        # 简单像素注意力
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')  # 深度可分离卷积
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 通道注意力
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化到1x1
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),  # 1x1卷积
            nn.Sigmoid()  # Sigmoid激活函数
        )

        # 像素注意力
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),  # 1x1卷积，降维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),  # 1x1卷积，输出单通道
            nn.Sigmoid()  # Sigmoid激活函数
        )

        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),  # 1x1卷积，升维
            nn.GELU(),  # GELU激活函数
            nn.Conv2d(dim * 4, dim, 1)  # 1x1卷积，降维
        )

    def forward(self, x):
        identity = x  # 保存输入以便残差连接
        x = self.norm2(x)  # 批归一化
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)  # 拼接不同注意力机制的输出
        x = self.mlp2(x)  # 通过MLP层
        x = identity + x  # 残差连接
        return x


class GuidedMultiScaleFrequencyAttention(nn.Module):
    """改进版：中频引导的多尺度频域注意力"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 频域分解分支
        self.low_freq = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, channels // 8, 1),  # 减少通道数
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),  # 增强中频容量
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.high_freq = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 3, padding=1),  # 减少通道数
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 中频引导的交叉注意力 (核心改进)
        self.low_guidance = CrossChannelAttention(channels // 8, channels // 2)
        self.high_guidance = CrossChannelAttention(channels // 8, channels // 2)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels // 8 + channels // 2 + channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 提取三频带特征
        low = self.low_freq(x)
        mid = self.mid_freq(x)
        high = self.high_freq(x)

        # 中频引导调整低/高频特征 (核心改进)
        low_guided = self.low_guidance(low, mid)
        high_guided = self.high_guidance(high, mid)

        # 融合调整后的特征
        fused = torch.cat([low_guided, mid, high_guided], dim=1)
        return self.fusion(fused)


class CrossChannelAttention(nn.Module):
    """轻量级交叉通道注意力"""

    def __init__(self, target_channels, guide_channels):
        super().__init__()
        self.guide_transform = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(guide_channels, target_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, target, guide):
        """使用引导特征调整目标特征"""
        guidance_weights = self.guide_transform(guide)
        return target * (1 + guidance_weights)  # 增强式调整


class DynamicChannelAttention(nn.Module):
    """动态通道注意力 (借鉴DLKA的权重分配)"""

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
    """高性能残差块"""

    def __init__(self, main):
        super().__init__()
        self.main = main

    def forward(self, x):
        return x + self.main(x)


# 核心模块 ------------------------------------------------------
class MGMFE(nn.Module):
    def __init__(self, nc, expand=2):
        super().__init__()
        self.nc = nc

        # 1. 可学习频域滤波器 (借鉴LFT)
        self.freq_filter = nn.Parameter(torch.ones(1, nc, 1, 1))

        # 频域处理核心
        self.process_stages = Enhanced_Parallel_Attention(nc)

        # 强化门控机制：中频引导的多尺度频域注意力
        self.gate = nn.Sequential(
            GuidedMultiScaleFrequencyAttention(nc),  # <- 使用改进版
            DynamicChannelAttention(nc)
        )

        # Scharr算子保持不变
        self.register_buffer('scharr_x', torch.tensor([[-3, 0, 3],
                                                       [-10, 0, 10],
                                                       [-3, 0, 3]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('scharr_y', torch.tensor([[-3, -10, -3],
                                                       [0, 0, 0],
                                                       [3, 10, 3]], dtype=torch.float32).view(1, 1, 3, 3))

        # 2. 边缘增强参数改为通道级控制
        self.edge_alpha = nn.Parameter(torch.zeros(1, nc, 1, 1))

        # 相位保留参数
        self.phase_gamma = nn.Parameter(torch.tensor(1.0))

        # 频域残差连接
        self.freq_residual = nn.Conv2d(nc, nc, 1, bias=False)

    def compute_gradient(self, x):
        """使用Scharr算子计算梯度幅值"""
        grad_x = F.conv2d(x, self.scharr_x.repeat(self.nc, 1, 1, 1),
                          padding=1, groups=self.nc)
        grad_y = F.conv2d(x, self.scharr_y.repeat(self.nc, 1, 1, 1),
                          padding=1, groups=self.nc)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # FFT变换
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        # 3. 应用可学习频域滤波
        mag = mag * self.freq_filter

        mag_processed = self.process_stages(mag) + self.freq_residual(mag)

        # 多尺度频域门控
        gate_weight = self.gate(mag_processed)
        mag = mag_processed * (1 + gate_weight)  # 增强型残差连接

        # 边缘增强 (使用通道级控制)
        edge_grad = self.compute_gradient(x)
        edge_freq = torch.fft.rfft2(edge_grad, norm='backward')
        edge_mag = torch.abs(edge_freq)
        mag = mag + self.edge_alpha * edge_mag  # 通道级权重控制

        # 重建信号
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        # 逆FFT变换
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        # 相位保留残差连接
        return identity * self.phase_gamma + x_out


class Frequency_Domain(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.freq = MGMFE(channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # 通道注意力增强
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 16), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // 16), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        # 通道注意力
        att = self.channel_att(x)
        x = x * att
        # 频域处理
        x = self.norm(x)
        x = self.freq(x)
        # 残差连接
        return identity + self.gamma * x


