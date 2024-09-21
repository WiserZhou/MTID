import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ObservationConvEncoder(nn.Module):
    def __init__(self, input_channels, output_channels,ie_num=2):
        super(ObservationConvEncoder, self).__init__()
        self.ie_num = ie_num
        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1)

        # 初始化卷积核为近似恒等映射
        with torch.no_grad():
            self.conv1.weight.fill_(0)
            self.conv1.bias.fill_(0)
            self.conv2.weight.fill_(0)
            self.conv2.bias.fill_(0)
            self.conv1.weight[:, :, 1] = 1
            self.conv2.weight[:, :, 1] = 1
            
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
    
class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, output_dim,ie_num):
        super(ObservationEncoder, self).__init__()
        ie_num= 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim*ie_num),
            nn.ReLU(),
            nn.Linear(input_dim*ie_num, input_dim*ie_num),
            nn.ReLU(),
            nn.Linear(input_dim*ie_num, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

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
        
        self.module_kind = args.module_kind
        self.encoder_kind = args.encoder_kind

        if self.encoder_kind == 'linear':
            self.encoder = ObservationEncoder(input_dim, output_dim,args.ie_num)
        else:
            self.encoder = ObservationConvEncoder(input_dim,output_dim,args.ie_num)
        

        self.interpolator = LatentSpaceInterpolator(
            output_dim, block_num)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            output_dim, num_heads=8, num_layers=args.transformer_num) for _ in range(num_transformer_blocks)])

    def forward(self, x1, x2):

        # print(x1.shape)torch.Size([256, 1536])
        
        if self.module_kind == 'i':
            interpolated_frames = self.interpolator(x1,x2)
            return interpolated_frames
        elif self.module_kind == 'e+i':
            x1_encoded = self.encoder(x1)
            x2_encoded = self.encoder(x2)
            interpolated_frames = self.interpolator(x1_encoded, x2_encoded)
            return interpolated_frames
        elif self.module_kind == 'i+t':
            interpolated_frames = self.interpolator(x1, x2)
            transformer_input = interpolated_frames
            for transformer_block in self.transformer_blocks:
                transformer_input = transformer_block(transformer_input)
            output = transformer_input + interpolated_frames
            return output
        else:
            x1_encoded = self.encoder(x1)
            x2_encoded = self.encoder(x2)

            # x1_encoded = x1
            # x2_encoded = x2
            # print(x1_encoded.shape)

            # print(x1_encoded.shape)torch.Size([256, 1536, 1])
            interpolated_frames = self.interpolator(
                x1_encoded, x2_encoded)
            
            # print(interpolated_frames.shape)

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
