import torch
import torch.nn as nn
import utils


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

    def forward(self, x, t=None):
        return self.sampling(x)


class AttentionBlock(utils.Module):
    def __init__(
        self,
        channels,
        num_heads=1
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.attention = nn.Sequential(
            nn.GroupNorm(32, self.channels),
            nn.Conv1d(self.channels, self.channels * 3, 1),
            utils.QKVAttention(self.num_heads)
        )

    def forward(self, x, t=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.attention(x)
        return (x + h).reshape(b, c, *spatial)


class ResidualBlock(utils.Module):

    def __init__(
        self,
        channel_in,
        channel_out,
        time_channel=None,
        dropout=0
    ):
        super().__init__()

        # member variables
        self.channel_base = channel_in
        self.channel_out = channel_out

        # layer definition
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channel, self.channel_out)
        ) if time_channel is not None else None
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, self.channel_base),
            nn.SiLU(),
            nn.Conv2d(self.channel_base, self.channel_out, kernel_size=3, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, self.channel_out),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.channel_out, self.channel_out, kernel_size=3, padding=1)
        )

        # skip connection
        if self.channel_base == self.channel_out:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(self.channel_base, self.channel_out, kernel_size=1)

    def forward(self, x, t=None):
        '''
        :param[in]  x   torch.Tensor [batch_size, x_channels, height, width]
        :param[in]  t   torch.Tensor [bathc_size, t_embedding]
        '''
        h = self.conv1(x)
        if t is not None and self.time_emb is not None:
            h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return self.skip_connection(x) + h


class UNet(nn.Module):

    def __init__(
        self,
        channel_in,
        channel_out=None,
        channel_base=64,
        n_res_blocks=2,
        dropout=0,
        channel_mult={1, 2, 4, 8},
        attention_head=4
    ):

        super().__init__()

        # member variables
        self.channel_in = channel_in
        self.channel_out = channel_out or channel_in
        self.channel_base = channel_base
        self.n_res_blocks = n_res_blocks
        self.dropout = dropout

        # time embedding
        time_embedding_channel = channel_base * 4
        self.time_embedding = nn.Sequential(
            nn.Linear(self.channel_base, time_embedding_channel),
            nn.SiLU(),
            nn.Linear(time_embedding_channel, time_embedding_channel)
        )

        # input block
        self.input = utils.Sequential(
            nn.Conv2d(self.channel_in, self.channel_base, kernel_size=1)
        )
        channel_sequence = [channel_base]

        # temporary variables
        # ch: temporary input channel
        ch = self.channel_base

        # encoder module list
        self.encoder_block = nn.ModuleList()
        for l, mult in enumerate(channel_mult):
            for _ in range(n_res_blocks):
                self.encoder_block.append(ResidualBlock(ch, mult * self.channel_base, time_channel=time_embedding_channel, dropout=dropout))
                ch = mult * self.channel_base
                channel_sequence.append(ch)
            if l != len(channel_mult) - 1:
                self.encoder_block.append(SampleBlock(sampling_type="down"))
                channel_sequence.append(ch)

        # bottomneck
        self.bottom_block = utils.Sequential(
            ResidualBlock(ch, ch, time_channel=time_embedding_channel, dropout=dropout),
            AttentionBlock(ch, attention_head),
            ResidualBlock(ch, ch, time_channel=time_embedding_channel, dropout=dropout)
        )

        # decoder module list
        self.decoder_block = nn.ModuleList()
        for l, mult in reversed(list(enumerate(channel_mult))):
            for _ in range(n_res_blocks):
                self.decoder_block.append(ResidualBlock(ch + channel_sequence.pop(), mult * self.channel_base, time_channel=time_embedding_channel, dropout=dropout))
                ch = mult * self.channel_base
            if l > 0:
                self.decoder_block.append(
                    utils.Sequential(
                        ResidualBlock(ch + channel_sequence.pop(), mult * self.channel_base, time_channel=time_embedding_channel, dropout=dropout),
                        SampleBlock(sampling_type="up")
                    )
                )
                ch = mult * self.channel_base

        # output block
        self.output = nn.Sequential(
            nn.GroupNorm(32, self.channel_base),
            nn.SiLU(),
            nn.Conv2d(ch, self.channel_out, kernel_size=1)
        )

    def forward(self, x, t=None):
        t_emb = self.time_embedding(utils.time_embedding(t, self.channel_base)) if t is not None else None
        h = self.input(x, t)
        ht = [h]
        for module in self.encoder_block:
            h = module(h, t_emb)
            ht.append(h)
        h = self.bottom_block(h, t_emb)
        for module in self.decoder_block:
            h = torch.cat([h, ht.pop()], dim=1)
            h = module(h, t_emb)
        h = h.type(x.dtype)
        return self.output(h)
