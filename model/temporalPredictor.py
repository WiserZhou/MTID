import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)


# Assuming Conv1dBlock, Rearrange, SinusoidalPosEmb, Downsample1d, Upsample1d are predefined

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, context):
        x = einops.rearrange(x, 'b t c -> t b c')
        context = einops.rearrange(context, 'b t c -> t b c')
        attn_output, _ = self.multihead_attn(x, context, context)
        x = x + attn_output
        x = self.layer_norm(x)
        x = x + self.ffn(x)
        x = einops.rearrange(x, 't b c -> b t c')
        return x


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=3, num_heads=4):
        super().__init__()

        # Define a sequence of convolutional blocks
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size, if_zero=True)
        ])

        # Time embedding block
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # Residual connection
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

        # Cross attention block
        self.cross_attention = CrossAttention(out_channels, num_heads)

    def forward(self, x, t, context):
        # Forward pass with time embedding (for diffusion models)
        out = self.blocks[0](x) + self.time_mlp(t)   # for diffusion
        # Uncomment the following line for Noise and Deterministic Baselines
        # out = self.blocks[0](x)
        out = self.blocks[1](out)
        out = self.cross_attention(out, context)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        dim=256,
        dim_mults=(1, 2, 4),
        num_heads=4
    ):
        super().__init__()

        # Determine the dimensions at each stage
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim

        # Define the time embedding module (for diffusion models)
        self.time_mlp = nn.Sequential(    # should be removed for Noise and Deterministic Baselines
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Create downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(
                    dim_in, dim_out, embed_dim=time_dim, num_heads=num_heads),
                ResidualTemporalBlock(
                    dim_out, dim_out, embed_dim=time_dim, num_heads=num_heads),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]

        # Define the middle blocks
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, num_heads=num_heads)
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, num_heads=num_heads)

        # Create upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(
                    dim_out * 2, dim_in, embed_dim=time_dim, num_heads=num_heads),
                ResidualTemporalBlock(
                    dim_in, dim_in, embed_dim=time_dim, num_heads=num_heads),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        # Final convolutional block
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, context):
        # Rearrange input tensor dimensions
        x = einops.rearrange(x, 'b h t -> b t h')

        # Get the time embedding (for diffusion models)
        # Uncomment the following line for Noise and Deterministic Baselines
        # t = None
        t = self.time_mlp(time)   # for diffusion
        h = []

        # Forward pass through downsampling blocks
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, context)
            x = resnet2(x, t, context)
            h.append(x)
            x = downsample(x)

        # Forward pass through middle blocks
        x = self.mid_block1(x, t, context)
        x = self.mid_block2(x, t, context)

        # Forward pass through upsampling blocks
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, context)
            x = resnet2(x, t, context)
            x = upsample(x)

        # Final convolution and rearrange dimensions back
        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x
