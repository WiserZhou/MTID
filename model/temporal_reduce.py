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

# Define a residual block used in the Temporal Unet


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=3):
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

    def forward(self, x, t):
        # Forward pass with time embedding (for diffusion models)
        out = self.blocks[0](x) + self.time_mlp(t)   # for diffusion
        # Uncomment the following line for Noise and Deterministic Baselines
        # out = self.blocks[0](x)
        # print(out.shape)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

# Define the Temporal Unet model


#  transition_dim=args.action_dim + args.observation_dim + args.class_dim,
class TemporalUnet(nn.Module):
    def __init__(
        self,
        args,
        dim=256,
        dim_mults=(1, 2, 4),
    ):
        super().__init__()

        transition_dim = args.action_dim + args.observation_dim + args.class_dim

        # Determine the dimensions at each stage
        # transition_dim is the initial dimension (1659)
        # dim is the base dimension (256)
        # dim_mults is a list of multipliers ([1, 2, 4])
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]

        # dims will be a list combining transition_dim and the scaled values of dim
        # For example: [1659, 256, 512, 1024]

        # in_out will be a list of tuples representing pairs of consecutive dimensions
        # [(1659, 256), (256, 512), (512, 1024)]
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
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                # ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]

        # Define the middle blocks
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim)
        # self.mid_block2 = ResidualTemporalBlock(
        #     mid_dim, mid_dim, embed_dim=time_dim)

        # Create upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                # ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        # Final convolutional block
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time):
        # Rearrange input tensor dimensions
        x = einops.rearrange(x, 'b h t -> b t h')

        # Get the time embedding (for diffusion models)
        # Uncomment the following line for Noise and Deterministic Baselines
        # t = None
        t = self.time_mlp(time)   # for diffusion
        h = []

        # print("start-------------------------------")

        # Forward pass through downsampling blocks
        for resnet, downsample in self.downs:
            # print(x.shape)
            x = resnet(x, t)
            # print(x.shape)
            # x = resnet2(x, t)
            h.append(x)
            x = downsample(x)
            # print("up--------------")

        # print("middle-------------")

        # Forward pass through middle blocks
        # print(x.shape)
        x = self.mid_block1(x, t)
        # print(x.shape)
        # x = self.mid_block2(x, t)

        # print("middle-------------")

        # Forward pass through upsampling blocks
        for resnet, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            # print(x.shape)
            x = resnet(x, t)
            # print(x.shape)
            # x = resnet2(x, t)
            x = upsample(x)
            # print("down--------------")

        # print("final-----------------------")
        # Final convolution and rearrange dimensions back
        # print(x.shape)
        x = self.final_conv(x)
        # print(x.shape)
        x = einops.rearrange(x, 'b t h -> b h t')
        # print(x.shape)
        return x
