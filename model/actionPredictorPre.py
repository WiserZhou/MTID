import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# import clip
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoProcessor
# Image Encoder


class ObservationEncoder(nn.Module):
    def __init__(self, input_channels, output_channels,ie_num=2):
        super(ObservationEncoder, self).__init__()
        self.ie_num = ie_num
        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    # input shape (batch_size,observation_dim)
    def forward(self, x):
        
        # print(x.shape)

        x = x.unsqueeze(2)
        
        # print(x.shape)
        
        if self.ie_num == 2:
            x = F.relu(self.conv1(x))
            # print(x.shape)
            x = F.relu(self.conv2(x))
            # print(x.shape)
        else:
            RuntimeError('unvalid ie_num!')

        x = x.squeeze(2)

        return x


class ObservationEncoderByCLIP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ObservationEncoderByCLIP, self).__init__()

        # Assuming input_dim is the flattened image size (e.g., 32x32x15 = 1536)
        # and we need to reshape the input to a valid image size before passing to CLIP
        # Simplified calculation for demonstration
        self.input_height = int(input_dim ** 0.5)
        self.input_width = self.input_height
        self.input_channels = 3  # Assuming RGB images

        # Load the CLIP model and processor directly
        self.processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")

    # input shape (batch_size, observation_dim)
    def forward(self, flat_images):
        # Reshape the flat images to the correct image dimensions
        images = flat_images.view(-1, self.input_channels,
                                  self.input_height, self.input_width)

        # Preprocess images
        pixel_values = self.processor(
            images=images, return_tensors="pt").pixel_values

        # Encode images using CLIP
        with torch.no_grad():
            image_features = self.model(
                pixel_values=pixel_values).pooler_output

        return image_features


# Semantic Space Interpolation

class LatentSpaceInterpolator(nn.Module):
    def __init__(self, dimension_num, block_num):
        super(LatentSpaceInterpolator, self).__init__()
        self.block_num = block_num
        self.dimension_num = dimension_num
        # Introduce a linear layer to generate the alpha values dynamically for each frame
        self.alpha_generator = nn.Linear(dimension_num, block_num)
        # Use Sigmoid activation function to constrain the alpha values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        batch_size, *_ = x1.shape

        # Stack x1 and x2 along a new dimension
        # Shape: (2, batch_size, dimension_num)
        x_combined = torch.stack([x1, x2], dim=0)

        # Generate unique alpha values for each block
        # Shape: (batch_size, block_num)
        alphas = self.sigmoid(self.alpha_generator(
            torch.ones(batch_size, self.dimension_num).to(x1.device)))

        # Expand alphas to match the dimensionality of the input
        # Shape: (batch_size, block_num, 1)
        alphas = alphas.unsqueeze(-1)

        # Compute the interpolated frames using broadcasting
        # Shape: (batch_size, block_num, dimension_num)
        interpolated_frames = (
            1 - alphas) * x_combined[0].unsqueeze(1) + alphas * x_combined[1].unsqueeze(1)

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


class ActionPredictor(nn.Module):
    def __init__(self, args, input_dim, output_dim, dimension_num, block_num, num_transformer_blocks=1):
        super(ActionPredictor, self).__init__()

        self.encoder = ObservationEncoder(input_dim, output_dim,args.ie_num)


        self.interpolator = LatentSpaceInterpolator(
            output_dim, block_num)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            output_dim, num_heads=8, num_layers=args.transformer_num) for _ in range(num_transformer_blocks)])

        # self.residual_conv = nn.Conv1d(input_dim, output_dim, 1) \
        #     if input_dim != output_dim else nn.Identity()

        # self.ffn = nn.Sequential(
        #     nn.Linear(output_dim, dimension_num * 4),
        #     nn.Mish(),
        #     nn.Linear(dimension_num * 4, dimension_num)
        # )

#   ActionPredictor(
#   x[:, 0, self.args.action_dim + self.args.observation_dim:],
#   x[:, -1, self.args.action_dim + self.args.observation_dim:],
    def forward(self, x1, x2):

        # print(x1.shape)torch.Size([256, 1536])

        x1_encoded = self.encoder(x1)
        x2_encoded = self.encoder(x2)

        # x1_encoded = x1
        # x2_encoded = x2

        # print(x1_encoded.shape)torch.Size([256, 1536, 1])

        interpolated_frames = self.interpolator(
            x1_encoded, x2_encoded)

        # print(interpolated_frames.shape)  # torch.Size([256, 12, 1536])

        # shape (num_frames, batch_size, observation_dim)
        # num_frames, batch_size, observation_dim = interpolated_frames.shape
        transformer_input = interpolated_frames
        # print(transformer_input.shape)torch.Size([256, 12, 1536])

        for transformer_block in self.transformer_blocks:
            transformer_input = transformer_block(transformer_input)

        output = transformer_input + interpolated_frames
        # print(output.shape)torch.Size([256, 12, 1536])

        # output = self.ffn(output)
        # print(output.shape)torch.Size([256, 12, 256])

        # resnet connect
        return output
