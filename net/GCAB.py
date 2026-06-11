import torch
import torch.nn as nn
import torch.nn.functional as F


class PCSA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = in_channels // reduction

        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.sa_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_weight = self.ca_fc(self.ca_avg_pool(x))
        x_ca = x * ca_weight

        sa_weight = self.sa_conv(x)
        x_sa = x * sa_weight

        out = x_ca + x_sa
        return out


class GCAB(nn.Module):
    def __init__(self, channels=64, bias=True, init_grad_weight=0.1):
        super().__init__()
        self.grad_weight = nn.Parameter(torch.tensor(init_grad_weight))

        kernel_size, stride, padding = 3, 1, 1
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=bias),
        )

        self.attn = PCSA(channels)

        self.grad_conv = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

        nn.init.constant_(self.grad_conv[0].weight, 0)
        nn.init.constant_(self.grad_conv[2].weight, 0)


    def compute_gradient(self, x):
        scharr_x = torch.tensor([[-3, 0, 3],
                                 [-10, 0, 10],
                                 [-3, 0, 3]],
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        scharr_y = torch.tensor([[-3, -10, -3],
                                 [0, 0, 0],
                                 [3, 10, 3]],
                                dtype=torch.float32, device=x.device).view(1, 1, 3, 3)

        scharr_x = scharr_x.repeat(x.size(1), 1, 1, 1)
        scharr_y = scharr_y.repeat(x.size(1), 1, 1, 1)

        grad_x = F.conv2d(x, scharr_x, padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, scharr_y, padding=1, groups=x.size(1))

        return torch.cat([grad_x, grad_y], dim=1)

    def forward(self, x):
        identity = x

        x1 = x + self.res(x)
        x2 = x1 + self.res(x1)
        x3 = x2 + self.res(x2)
        x3_1 = x1 + x3
        x4 = x3_1 + self.res(x3_1)
        x4_1 = x + x4
        x5 = self.attn(x4_1)
        x5_1 = x + x5

        input_grad = self.compute_gradient(identity)
        output_grad = self.compute_gradient(x5_1)

        grad_feature = torch.cat([input_grad, output_grad], dim=1)
        grad_adjust = self.grad_conv(grad_feature)

        out = x5_1 + self.grad_weight * grad_adjust

        return out