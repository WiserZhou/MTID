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
from .actionPredictorPre import (
    MotionPredictor,
)

# Assuming Conv1dBlock, Rearrange, SinusoidalPosEmb, Downsample1d, Upsample1d are predefined


class CrossAttention(nn.Module):
    def __init__(self, observation_dim, embed_dim, num_heads,interpolation_num):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.linear = nn.Linear(observation_dim*interpolation_num, embed_dim)

    def forward(self, x, context):
        # print(x.shape)  # torch.Size([256, 256, 4])
        # print(context.shape)  # torch.Size([256, 7680])
        context = context.unsqueeze(2)  # torch.Size([256,7680,1])
        x = einops.rearrange(x, 'b t c -> b c t') 
        context = einops.rearrange(context, 'b s c -> b c s')
        # print(x.shape) # torch.Size([256,4,  256])
        # print(context.shape)  # torch.Size([256,1,  7680])
        context = self.linear(context)
        # print(x.shape)  # torch.Size([256,4,  256])
        # print(context.shape)  # torch.Size([256,1,256])
        attn_output, _ = self.multihead_attn(x, context, context)
        x = x + attn_output
        x = self.layer_norm(x)
        x = x + self.ffn(x)
        x = einops.rearrange(x, 'b c t -> b t c')
        return x  # torch.Size([256, 4, 256])


class ResidualTemporalBlock(nn.Module):

    def __init__(self, observation_dim, inp_channels, out_channels,interpolation_num, embed_dim, kernel_size=3, num_heads=4):
        super().__init__()
        
        self.out_channels = out_channels

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
        self.cross_attention = CrossAttention(
            observation_dim, out_channels, num_heads,interpolation_num)

    def forward(self, x, t, context):

        # Forward pass with time embedding (for diffusion models)
        out = self.blocks[0](x) + self.time_mlp(t)   # for diffusion

        # print(out.shape)  # torch.Size([256, 256, 4])
        out2 = self.cross_attention(out, context)
        # print(out2.shape) # torch.Size([256, 4, 256])

        out = out2 + out

        out = self.blocks[1](out)
        
        # print("residual",out.shape)
        # print('self.out_channels',self.out_channels)

        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        args,
        dim=256,
        dim_mults=(1, 2, 4),
        num_heads=4
    ):
        super().__init__()

        self.args = args

        transition_dim = args.action_dim + args.observation_dim + args.class_dim
        self.interpolation_num = 5

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
        self.block_num = 0

        # Create downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(args.observation_dim,
                                      dim_in, dim_out,self.interpolation_num, embed_dim=time_dim, num_heads=num_heads),
                ResidualTemporalBlock(args.observation_dim,
                                      dim_out, dim_out,self.interpolation_num, embed_dim=time_dim, num_heads=num_heads),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

            self.block_num += 2

        mid_dim = dims[-1]

        # Define the middle blocks
        self.mid_block1 = ResidualTemporalBlock(args.observation_dim,
                                                mid_dim, mid_dim,self.interpolation_num, embed_dim=time_dim, num_heads=num_heads)
        self.mid_block2 = ResidualTemporalBlock(args.observation_dim,
                                                mid_dim, mid_dim,self.interpolation_num, embed_dim=time_dim, num_heads=num_heads)

        self.block_num += 2

        # Create upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(args.observation_dim,
                                      dim_out * 2, dim_in,self.interpolation_num, embed_dim=time_dim, num_heads=num_heads),
                ResidualTemporalBlock(args.observation_dim,
                                      dim_in, dim_in,self.interpolation_num, embed_dim=time_dim, num_heads=num_heads),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            self.block_num += 2

        # Final convolutional block
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )
        
        
        self.motionPredictor = MotionPredictor(self.args,
                                               self.args.observation_dim,
                                               self.args.observation_dim,
                                               dim,
                                               self.interpolation_num
                                               )

    # x shape (batch_size,horizon,dimension)


    def forward(self, x, time):

        # print(x.shape)torch.Size([256, 3, 1659])
        # print(time.shape)torch.Size([256])
        #  args.action_dim + args.observation_dim + args.class_dim

        # shape (num_frames, batch_size, observation_dim)
        cross_features = self.motionPredictor(x[:, 0, self.args.class_dim + self.args.action_dim:],
                                              x[:, -1, self.args.class_dim + self.args.action_dim:])

        # print(cross_features.shape) # torch.Size([256, 5, 1536])
        # cross_features = cross_features.permute(1, 0, 2)
        # cross_features_pre1 = cross_features[:,self.interpolation_num//2,:].repeat(1,self.interpolation_num,1)
        # cross_features_pre1 = cross_features_pre1.view(cross_features.shape[0],-1)
        
        # print(cross_features_pre1.shape) # torch.Size([256, 7680])
        
        # cross_features_pre2 = cross_features[:,1:4,:].repeat(1,2,1)
        # cross_features_pre2 = cross_features_pre2.view(cross_features.shape[0],-1)       
        
        
        cross_features = cross_features.view(cross_features.shape[0],-1) # torch.Size([256, 7680])
        
        # print(cross_features.shape) torch.Size([256, 1536])
        if cross_features.shape[1] != self.args.observation_dim * self.interpolation_num:
            batch_size = cross_features.size(0)
            padding = torch.zeros(batch_size, 
                                    self.args.observation_dim* self.interpolation_num - cross_features.shape[1]
                                    ).to(cross_features.device) 
            return torch.cat([cross_features, padding], dim=1)
        # x shape (batch_size,horizon,dimension)
        # Rearrange input tensor dimensions
        x = einops.rearrange(x, 'b h t -> b t h')

        # Get the time embedding (for diffusion models)
        # Uncomment the following line for Noise and Deterministic Baselines
        # t = None

        # t = torch.randint(0, self.n_timesteps, (batch_size,),
        #    device=x.device).long()  # Random timestep for diffusion
        t = self.time_mlp(time)   # for diffusion

        # print(t.shape)torch.Size([256, 256])
        h = []
        index = 0

        # Forward pass through downsampling blocks
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, cross_features)
            index += 1
            x = resnet2(x, t, cross_features)
            index += 1
            h.append(x)
            x = downsample(x)

        # Forward pass through middle blocks
        x = self.mid_block1(x, t, cross_features)
        index += 1
        x = self.mid_block2(x, t, cross_features)
        index += 1

        # Forward pass through upsampling blocks
        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, cross_features)
            index += 1
            x = resnet2(x, t, cross_features)
            index += 1
            x = upsample(x)

        # Final convolution and rearrange dimensions back

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        


        return x 
        # return x



# shape of x

# torch.Size([12, 256, 1536])

# start-------------------------------
# torch.Size([256, 256, 3])
# torch.Size([256, 256, 3])
# up--------------
# torch.Size([256, 512, 2])
# torch.Size([256, 512, 2])
# up--------------
# torch.Size([256, 1024, 1])
# torch.Size([256, 1024, 1])
# up--------------
# middle-------------
# torch.Size([256, 1024, 1])
# torch.Size([256, 1024, 1])
# middle-------------
# torch.Size([256, 512, 1])
# torch.Size([256, 512, 1])
# down--------------
# torch.Size([256, 256, 2])
# torch.Size([256, 256, 2])
# down--------------
# final-----------------------
# torch.Size([256, 256, 3])
# torch.Size([256, 1659, 3])
# torch.Size([256, 3, 1659])
