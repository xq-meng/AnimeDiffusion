import torch
import torch.nn as nn


class ChannelAttention(nn.Module):

    def __init__(self, in_channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, False),
            nn.ReLU(),
            nn.Linear(in_channel // ratio, in_channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out = self.max_pool(x).view(b, c)
        avg_pool_out = self.avg_pool(x).view(b, c)

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)

        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return x * out


class CBAM(nn.Module):

    def __init__(self, in_channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x