import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import einops
from einops.layers.torch import Rearrange

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
# Image Encoder


class ImageEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Semantic Space Interpolation


class SemanticSpaceInterpolation(nn.Module):
    def __init__(self, num_frames):
        super(SemanticSpaceInterpolation, self).__init__()
        # Introduce a linear layer to generate the alpha values dynamically for each frame
        self.alpha_generator = nn.Linear(1, num_frames)
        # Use ReLU activation function to ensure the alpha values are positive
        self.relu = nn.ReLU()
        # Use Sigmoid activation function to constrain the alpha values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, num_frames):
        batch_size, channels, height, width = x1.shape

        # Generate an alpha matrix of shape (batch_size, num_frames)
        alpha_matrix = self.alpha_generator(
            torch.ones(batch_size, 1).to(x1.device))
        # Ensure alpha values are positive
        alpha_matrix = self.relu(alpha_matrix)
        # Constrain alpha values between 0 and 1
        alpha_matrix = self.sigmoid(alpha_matrix)

        # Expand the alpha matrix to (batch_size, num_frames, channels, height, width)
        alpha_matrix = alpha_matrix.view(batch_size, num_frames, 1, 1, 1)

        # Duplicate x1 and x2 to match the shape of alpha_matrix
        x1_repeated = x1.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        x2_repeated = x2.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        # Compute the interpolated frames
        interpolated_frames = (1 - alpha_matrix) * \
            x1_repeated + alpha_matrix * x2_repeated

        return interpolated_frames

# Transformer Block


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_layers):
        super(TransformerBlock, self).__init__()
        encoder_layers = TransformerEncoderLayer(dim, num_heads)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        x = self.transformer(x)
        return x
# Motion Predictor with Transformer blocks


class MotionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_transformer_blocks):
        super(MotionPredictor, self).__init__()
        self.encoder = ImageEncoder(input_dim, hidden_dim)
        self.interpolator = SemanticSpaceInterpolation()
        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            hidden_dim, num_heads=8, num_layers=6) for _ in range(num_transformer_blocks)])
        self.unet = TemporalUnet(
            hidden_dim, output_dim, num_transformer_blocks)

    def forward(self, x1, x2, num_frames):
        x1_encoded = self.encoder(x1)
        x2_encoded = self.encoder(x2)
        interpolated_frames = self.interpolator(
            x1_encoded, x2_encoded, num_frames)

        batch_size, num_frames, channels, height, width = interpolated_frames.shape
        transformer_input = interpolated_frames.view(
            batch_size, num_frames, -1)

        cross_attentions = []
        for transformer_block in self.transformer_blocks:
            transformer_output = transformer_block(transformer_input)
            cross_attentions.append(transformer_output.view(
                batch_size, num_frames, channels, height, width))

        outputs = []
        for i in range(num_frames):
            unet_output = self.unet(interpolated_frames[:, i], [
                                    cross_attentions[j][:, i] for j in range(len(cross_attentions))])
            outputs.append(unet_output)

        return torch.stack(outputs, dim=1)

# Transition Video Diffusion Model


class TransitionVideoDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_transformer_blocks):
        super(TransitionVideoDiffusionModel, self).__init__()
        self.motion_predictor = MotionPredictor(
            input_dim, hidden_dim, output_dim, num_transformer_blocks)

    def forward(self, x1, x2, num_frames):
        return self.motion_predictor(x1, x2, num_frames)


# Define a residual block used in the Temporal Unet


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size, if_zero=True)
        ])
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.residual_conv = nn.Conv1d(
            inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cross_attention=None):
        if cross_attention is not None:
            x = x + cross_attention
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

# Define the Temporal Unet model


class TemporalUnet(nn.Module):
    def __init__(self, transition_dim, dim=256, dim_mults=(1, 2, 4)):
        super().__init__()
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=3, if_zero=True),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, time, cross_attentions):
        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)
        h = []
        attention_index = 0

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, cross_attention=cross_attentions[attention_index])
            attention_index += 1
            x = resnet2(
                x, t, cross_attention=cross_attentions[attention_index])
            attention_index += 1
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(
            x, t, cross_attention=cross_attentions[attention_index])
        attention_index += 1
        x = self.mid_block2(
            x, t, cross_attention=cross_attentions[attention_index])
        attention_index += 1

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, cross_attention=cross_attentions[attention_index])
            attention_index += 1
            x = resnet2(
                x, t, cross_attention=cross_attentions[attention_index])
            attention_index += 1
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x


# Example usage
if __name__ == "__main__":
    input_dim = 3  # RGB images
    hidden_dim = 128
    output_dim = 3  # Output RGB images
    num_frames = 8  # Number of frames to generate
    num_transformer_blocks = 2  # Number of transformer blocks

    model = TransitionVideoDiffusionModel(
        input_dim, hidden_dim, output_dim, num_transformer_blocks)

    example_image1 = torch.randn(2, 3, 64, 64)
    example_image2 = torch.randn(2, 3, 64, 64)

    output = model(example_image1, example_image2, num_frames)
    print(output.shape)  # Should print (2, 8, 3, 64, 64)
