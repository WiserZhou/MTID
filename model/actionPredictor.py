import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import clip
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel
# Image Encoder


class ImageEncoder(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ImageEncoder, self).__init__()
        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1)

    # input shape (batch_size,observation_dim)
    def forward(self, x):

        x = x.unsqueeze(2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.squeeze(2)

        return x


class ImageEncoderByCLIP(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load the CLIP model directly
        self.model, _ = AutoModel.from_pretrained(
            "/home/zhouyufan/Projects/PDPP/dataset/ViT-B-32__openai")
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    # input shape (batch_size,observation_dim)
    def forward(self, images):
        # Assume images are already loaded and in the form of (batch_size, height, width, channels)
        # Convert images to (batch_size, channels, height, width)
        images = images.permute(0, 3, 1, 2)

        # Preprocess images
        images = torch.stack([self.preprocess(image) for image in images])

        # Encode images using CLIP
        with torch.no_grad():
            image_features = self.model.encode_image(images)

        return image_features
# Semantic Space Interpolation


class SemanticSpaceInterpolation(nn.Module):
    def __init__(self, dimension_num, block_num):
        super(SemanticSpaceInterpolation, self).__init__()
        self.block_num = block_num
        self.dimension_num = dimension_num
        # Introduce a linear layer to generate the alpha values dynamically for each frame
        self.alpha_generator = nn.Linear(
            self.dimension_num, self.dimension_num)
        # Use ReLU activation function to ensure the alpha values are positive
        self.relu = nn.ReLU()
        # Use Sigmoid activation function to constrain the alpha values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        # print(x1.shape)torch.Size([256, 1536, 1])

        batch_size, *_ = x1.shape

        # Stack x1 and x2 along a new dimension
        # Shape: (2, batch_size, observation_dim)
        x_combined = torch.stack([x1, x2], dim=0)

        # print(x_combined.shape)torch.Size([2, 256, 1536])

        # Generate an alpha matrix of shape (batch_size, dimension_num)
        alpha_matrix = self.alpha_generator(
            torch.ones(batch_size, self.dimension_num).to(x1.device))

        # Ensure alpha values are positive
        alpha_matrix = self.relu(alpha_matrix)
        # Constrain alpha values between 0 and 1
        alpha_matrix = self.sigmoid(alpha_matrix)

        # Expand the alpha matrix to (batch_size, dimension_num)
        alpha_matrix = alpha_matrix.view(batch_size, self.dimension_num)

        # print(alpha_matrix.shape)torch.Size([256, 1536])
        # Compute the interpolated frames
        interpolated_frames = (
            1 - alpha_matrix) * x_combined[0].unsqueeze(0) + alpha_matrix * x_combined[1].unsqueeze(0)

        # print(interpolated_frames.shape)torch.Size([1, 256, 1536])
        interpolated_frames = interpolated_frames.repeat_interleave(
            self.block_num, dim=0)

        # print(interpolated_frames.shape)torch.Size([12, 256, 1536])
        # Rearrange dimensions to (dimension_num, batch_size, observation_dim)
        interpolated_frames = interpolated_frames.permute(1, 0, 2)

        # print(interpolated_frames.shape)torch.Size([256, 12, 1536])
        # torch.Size([256, 12, 1536])
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
    def __init__(self, args, input_dim, output_dim, dimension_num, block_num, num_transformer_blocks=1):
        super(MotionPredictor, self).__init__()
        if args.debug == 0:
            self.encoder = ImageEncoder(input_dim, output_dim)
        elif args.debug == 2:
            self.encoder == ImageEncoderByCLIP(input_dim, output_dim)
        else:
            print("ERROR!")
        self.interpolator = SemanticSpaceInterpolation(
            output_dim, block_num)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            output_dim, num_heads=8, num_layers=args.transformer_num) for _ in range(num_transformer_blocks)])

        self.residual_conv = nn.Conv1d(input_dim, output_dim, 1) \
            if input_dim != output_dim else nn.Identity()

        # self.ffn = nn.Sequential(
        #     nn.Linear(output_dim, dimension_num * 4),
        #     nn.Mish(),
        #     nn.Linear(dimension_num * 4, dimension_num)
        # )

#   MotionPredictor(
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
