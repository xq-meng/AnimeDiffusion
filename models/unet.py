import torch
import torch.nn as nn
import torch.nn.functional as F


class SampleBlock(nn.Module):

    def __init__(
        self,
        sampling_type=None,
    ):
        super().__init__()

        # member variables
        if sampling_type == "up":
            self.sampling = nn.Upsample(scale_factor=2, mode="nearest")
        elif sampling_type == "down":
            self.sampling = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.sampling = nn.Identity()

    def forward(self, x):
        return self.sampling(x)


class ResidualBlock(nn.Module):

    def __init__(
        self,
        channel_in,
        channel_out=None,
        dropout=0
    ):
        super().__init__()

        # member variables
        self.channel_base = channel_in
        self.channel_out = channel_out or channel_in

        # layer definition
        self.conv = nn.Sequential(
            nn.GroupNorm(32, self.channel_base),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.channel_base, self.channel_out, kernel_size=3, padding=1)
        )

        # skip connection
        if self.channel_base == self.channel_out:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(self.channel_base, self.channel_out, kernel_size=1)

    def forward(self, x):
        h = self.conv(x)
        return self.skip_connection(x) + h


class UNet(nn.Module):

    def __init__(
        self,
        image_size,
        channel_in,
        channel_out=None,
        channel_base=64,
        n_res_blocks=2,
        dropout=0,
        depth=4
    ):

        super().__init__()

        # member variables
        self.image_size = image_size
        self.channel_in = channel_in
        self.channel_out = channel_out or channel_in
        self.channel_base = channel_base
        self.n_res_blocks = n_res_blocks
        self.dropout = dropout

        # temporary variables
        # tchi: temporary input channel
        # tcho: temporary output channel
        tchi = self.channel_in
        tcho = self.channel_base

        # input block 
        self.input = nn.Sequential(
            nn.Conv2d(tchi, tcho, kernel_size=1)
        )

        # encoder module list
        tchi = tcho
        self.encoder_block = nn.ModuleList()
        for l in range(depth):
            tlayer = nn.Sequential()
            if l != 0:
                tlayer.add_module(
                    'encoder_dsp_{0}'.format(l), SampleBlock(sampling_type="down")
                )
            for _ in range(n_res_blocks):
                tlayer.add_module(
                    'encoder_res_{0}_{1}'.format(l, _), ResidualBlock(tchi, tcho, dropout=dropout)
                )
                tchi = tcho
            self.encoder_block.append(tlayer)
            tcho = tcho * 2

        # bottomneck
        self.bottom_block = nn.Sequential()
        self.bottom_block.add_module(
            'bottom_neck_dsp_0', SampleBlock(sampling_type="down")
        )
        for _ in range(n_res_blocks):
            self.bottom_block.add_module(
                'bottom_neck_res_{0}'.format(_), ResidualBlock(tchi, tcho, dropout=dropout)
            )
            tchi = tcho
        self.bottom_block.add_module(
            'bottom_neck_usp_0', SampleBlock(sampling_type="up")
        )
        self.bottom_block.add_module(
            'bottom_neck_usp_conv_0', nn.Conv2d(tchi, tchi // 2, kernel_size=3, padding=1)
        )

        # decoder module list
        tcho = tchi // 2
        self.decoder_block = nn.ModuleList()
        for l in range(depth):
            tlayer = nn.Sequential()
            for _ in range(n_res_blocks):
                tlayer.add_module(
                    'decoder_res_{0}_{1}'.format(l, _), ResidualBlock(tchi, tcho, dropout=dropout)
                )
                tchi = tcho
            if l != depth - 1:
                tlayer.add_module(
                    'decoder_usp_{0}'.format(l), SampleBlock(sampling_type="up")
                )
                tlayer.add_module(
                    'decoder_usp_conv_{0}'.format(l), nn.Conv2d(tchi, tchi // 2, kernel_size=3, padding=1)
                )
            self.decoder_block.append(tlayer)
            tcho = tcho // 2

        # output block
        self.output = nn.Sequential(
            nn.GroupNorm(32, self.channel_base),
            nn.SiLU(),
            nn.Conv2d(tchi, self.channel_out, kernel_size=1)
        )


    def forward(self, x):
        ht = []
        h = self.input(x)
        for module in self.encoder_block:
            h = module(h)
            ht.append(h)
        h = self.bottom_block(h)
        for module in self.decoder_block:
            h = torch.cat([h, ht.pop()], dim=1)
            h = module(h)
        h = h.type(x.dtype)
        return self.output(h)


if __name__ == '__main__':
    b, c, h = 3, 6, 64
    model = UNet(image_size=h, channel_in=c)
    x = torch.randn((b, c, h, h))
    h = model(x)