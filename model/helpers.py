import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
import logging
from tensorboardX import SummaryWriter
import itertools
from collections import OrderedDict
# from loss_function import compute_losses
from utils.args import get_args

# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SinusoidalPosEmb(nn.Module):
    """
    SinusoidalPosEmb is a module that generates sinusoidal positional embeddings.
    These embeddings are commonly used in sequence models like Transformers to 
    provide the model with information about the positions of elements in the sequence.
    """

    def __init__(self, dim):
        """
        Initialize the SinusoidalPosEmb module.

        Parameters:
        - dim (int): The dimension of the positional embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Forward pass to generate the positional embeddings.

        Parameters:
        - x (Tensor): A tensor containing the positions for which to generate embeddings.

        Returns:
        - Tensor: A tensor containing the sinusoidal positional embeddings.
        """
        # Get the device of the input tensor (CPU or GPU)
        device = x.device

        # Calculate half the dimension, as we will create embeddings for both sine and cosine
        half_dim = self.dim // 2

        # Calculate the scaling factor for the embedding
        # We use log(10000) to spread out the positional embeddings in a range
        emb = math.log(10000) / (half_dim - 1)

        # Create a tensor of size (half_dim) with values exponentially scaled by the factor calculated above
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Expand the dimensions of x and emb for broadcasting
        # x[:, None] changes shape from (batch_size) to (batch_size, 1)
        # emb[None, :] changes shape from (half_dim) to (1, half_dim)
        # This allows element-wise multiplication to create a (batch_size, half_dim) tensor
        emb = x[:, None] * emb[None, :]

        # Calculate the sine and cosine of the embedding values
        # Concatenate these values along the last dimension to create the final embeddings
        # Resulting shape will be (batch_size, dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class FourierPosEmb(nn.Module):
    """
    FourierPosEmb is a module that generates Fourier positional embeddings.
    These embeddings are used to provide the model with information about 
    the positions of elements in the sequence using Fourier features.
    """

    def __init__(self, dim, scale=1.0):
        """
        Initialize the FourierPosEmb module.

        Parameters:
        - dim (int): The dimension of the positional embeddings.
        - scale (float): The scaling factor for the random projection.
        """
        super().__init__()
        self.dim = dim
        self.scale = scale
        # Initialize the random projection matrix
        self.proj = nn.Parameter(torch.randn(
            1, dim) * scale, requires_grad=False)

    def forward(self, x):
        """
        Forward pass to generate the positional embeddings.

        Parameters:
        - x (Tensor): A tensor containing the positions for which to generate embeddings.

        Returns:
        - Tensor: A tensor containing the Fourier positional embeddings.
        """
        # Get the device of the input tensor (CPU or GPU)
        device = x.device

        x = x.float()

        # Project the input positions using the random projection matrix
        x_proj = x[:, None] @ self.proj.to(device)

        # Calculate the sine and cosine of the projected values
        # Concatenate these values along the last dimension to create the final embeddings
        # Resulting shape will be (batch_size, dim)
        emb = torch.cat((x_proj.sin(), x_proj.cos()), dim=-1)

        return emb

# RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x512 and 256x1024)
# nohup python main_distributed.py --layer_num=5 --name=5layer --gpu=1 > out/output_5layer.log 2>&1


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv = nn.Conv1d(dim, dim, 2, 1, 0)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 2, 1, 0)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    A block consisting of Conv1d, GroupNorm, and Mish activation.
    Optionally includes Dropout and can use zero-initialized weights.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=32, drop_out=0.0, if_zero=False):
        super().__init__()

        # Define the block with dropout, if specified
        if drop_out > 0.0:
            self.block = nn.Sequential(
                zero_module(  # Use zero-initialized weights if specified
                    nn.Conv1d(inp_channels, out_channels,
                              kernel_size, padding=1),
                ),
                # Rearrange the dimensions to fit GroupNorm's expected input shape
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),  # Apply Mish activation function
                # Apply Dropout with specified probability
                nn.Dropout(p=drop_out),
            )
        # Define the block with zero-initialized weights, if specified
        elif if_zero:
            self.block = nn.Sequential(
                zero_module(
                    nn.Conv1d(inp_channels, out_channels,
                              kernel_size, padding=1),
                ),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )
        # Define the standard block without Dropout and without zero-initialized weights
        else:
            self.block = nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )

    def forward(self, x):
        # Forward pass through the block
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#

# self.sqrt_alphas_cumprod, t, x_start.shape
def extract(a, t, x_shape):
    # Extract values from tensor `a` based on the indices provided in tensor `t`.
    # Reshape the output to match the dimensions required for broadcasting with `x_shape`.

    # Get the batch size from the shape of tensor `t`.
    # The batch size is the first dimension of `t`.
    batch_size, *_ = t.shape

    # Gather values from `a` using the indices from `t`.
    # The -1 in the gather method indicates gathering along the last dimension of `a`.
    out = a.gather(-1, t)

    # Reshape the gathered output to ensure it is compatible with the target shape `x_shape`.
    # The reshaping step is critical for broadcasting in further operations.
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine condition_projection
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def condition_projection(x, conditions, action_dim, class_dim):

    # clone the observation dim at the start and end
    for t, val in conditions.items():
        if t != 'task':
            x[:, t, class_dim + action_dim:] = val.clone()

    # def mask(x[:, :, :class_dim], x[:, :, class_dim:class_dim+action_dim]):
    #     task_class = find(x[:, :, :class_dim])
    #     action_one_hot = transform_find(task_class)
    #     x[:, :, class_dim:class_dim+action_dim][action_one_hot] = 0

    # set the observation to zero except for start and end
    x[:, 1:-1, class_dim + action_dim:] = 0.
    x[:, :, :class_dim] = conditions['task']

    return x

def compute_mask(x):
    args = get_args()
    task_class = {
        "0": 23521,
        "1": 59684,
        "2": 71781,
        "3": 113766,
        "4": 105222,
        "5": 94276,
        "6": 53193,
        "7": 105253,
        "8": 44047,
        "9": 76400,
        "10": 16815,
        "11": 95603,
        "12": 109972,
        "13": 44789,
        "14": 40567,
        "15": 77721,
        "16": 87706,
        "17": 91515,
    }
    task_ids = torch.argmax(x[:, :, : args.class_dim], axis=-1)  # (256*3)
    task_ids = task_ids[:, 0]
    action_one_hot = np.load(
        os.path.join(
            "/data/zhaobo/zhouyufan/PDPP-Optimize/dataset/crosstask",
            "crosstask_release",
            "actions_one_hot.npy",
        ),
        allow_pickle=True,
    ).item()

    def find_action_index(task_id, action_one_hot):
        action_indices = []
        for key, value in action_one_hot.items():
            # 检查键是否以 task_id 开头
            if key.startswith(str(task_id) + "_"):
                action_index = value
                action_indices.append(action_index)
        return action_indices

    mask = torch.zeros(x.shape[0], args.action_dim)
    for i in range(len(task_ids)):
        task_id = str(task_ids[i].item())
        action_indices = find_action_index(task_class[task_id], action_one_hot)
        for j in action_indices:
            mask[i, j] = 1.0

    maskTotal = torch.ones_like(x)
    maskTotal[:, :, args.class_dim: args.class_dim + args.action_dim] = mask.unsqueeze(
        1
    ).repeat(1, args.horizon, 1)
    return maskTotal.cuda()

# -----------------------------------------------------------------------------#
# ---------------------------------- Loss -------------------------------------#
# -----------------------------------------------------------------------------#

from model.loss_function import compute_losses
class Sequence_CE(nn.Module):

    def __init__(self,  action_dim, class_dim, weight):
        super().__init__()
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.weight = weight

    def forward(self, pred, targ,l_order=200.0, l_pos=0.01, l_perm=2.0,kind = 0):
        """
        :param pred: [B, T, task_dim+action_dim+observation_dim]
        :param targ: [B, T, task_dim+action_dim+observation_dim]
        :return:
        """
        mse_loss = F.mse_loss(pred, targ, reduction='none').sum()
        
        total_loss, ce_loss, order_loss, pos_loss, \
            perm_loss = compute_losses(pred[:,:,self.class_dim:self.class_dim +
                                        self.action_dim], targ[:,:,self.class_dim:self.class_dim +
                                        self.action_dim], lambda_oc=l_order,
                                        lambda_fp=l_pos, lambda_r=l_perm)
        if kind == 0 :
            return ce_loss
        elif kind == 1:
            return ce_loss+order_loss
        elif kind == 2:
            return ce_loss + pos_loss
        elif kind == 3:
            return ce_loss + perm_loss
        elif kind == 4:
            return mse_loss
        elif kind == 5:
            return mse_loss + order_loss
        elif kind == 6:
            return mse_loss + pos_loss
        elif kind == 7:
            return mse_loss + perm_loss
        else:
            RuntimeError('unvalid kind')
        
        def scale_tuple_elements(t, factor=1000):
            return tuple(x * factor for x in t)
        
        # loss_action = scale_tuple_elements(loss_action)
        # print(loss_action)

        return ce_loss


class Weighted_Gradient_MSE(nn.Module):

    def __init__(self,  action_dim, class_dim, weight):
        super().__init__()
        # self.register_buffer('weights', weights)
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.weight = weight

    def forward(self, pred, targ, mask=None):
        """
        :param pred: [B, T, task_dim+action_dim+observation_dim]
        :param targ: [B, T, task_dim+action_dim+observation_dim]
        :return:
        """

        loss_action = F.mse_loss(pred, targ, reduction='none')
        # Gradient weight
        batch_size, time_steps, _ = pred.size()
        half_time_steps = time_steps // 2
        if time_steps % 2 == 0:
            weights = torch.linspace(
                self.weight, 1, half_time_steps, device=pred.device)
            weights = torch.cat([weights, weights.flip(0)])
        else:
            weights = torch.cat([
                torch.linspace(self.weight, 1, half_time_steps +
                               1, device=pred.device),
                torch.linspace(1, self.weight, half_time_steps +
                               1, device=pred.device)[1:]
            ])


        if mask is not None:
            # Scale tensor
            scale = torch.full((time_steps,), 1.1, device=pred.device)
            # Apply weights to the action part of the loss
            loss_action = loss_action[
                :, :, self.class_dim: self.class_dim + self.action_dim
            ]
            mask = mask[:, :, self.class_dim: self.class_dim + self.action_dim]
            for t in range(time_steps):
                loss_action[:, t, :] = torch.where(
                    mask[:, t, :] == 1,
                    loss_action[:, t, :] * scale[t],
                    loss_action[:, t, :],
                )
                loss_action[:, t, :] *= weights[t]

        else:
            for t in range(time_steps):
                loss_action[:, t, self.class_dim:self.class_dim +
                            self.action_dim] *= weights[t]

        loss_action = loss_action.sum()
        
        # print(loss_action)
        return loss_action
    
def variance_loss(predictions):
    mean = torch.mean(predictions)
    variance = torch.mean((predictions - mean) ** 2)
    return variance.sum()
class Variance_Weighted_MSE(nn.Module):

    def __init__(self,  action_dim, class_dim, weight):
        super().__init__()
        # self.register_buffer('weights', weights)
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.weight = weight
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, pred, targ):
        """
        :param pred: [B, T, task_dim+action_dim+observation_dim]
        :param targ: [B, T, task_dim+action_dim+observation_dim]
        :return:
        """
        variance_loss_action = variance_loss(pred)

        loss_action = F.mse_loss(pred, targ, reduction='none')
        # Gradient weight
        batch_size, time_steps, _ = pred.size()
        half_time_steps = time_steps // 2
        if time_steps % 2 == 0:
            weights = torch.linspace(
                self.weight, 1, half_time_steps, device=pred.device)
            weights = torch.cat([weights, weights.flip(0)])
        else:
            weights = torch.cat([
                torch.linspace(self.weight, 1, half_time_steps +
                               1, device=pred.device),
                torch.linspace(1, self.weight, half_time_steps +
                               1, device=pred.device)[1:]
            ])

        # Apply weights to the action part of the loss
        for t in range(time_steps):
            loss_action[:, t, self.class_dim:self.class_dim +
                        self.action_dim] *= weights[t]

        loss_action = loss_action.sum()
        return loss_action



# 示例用法
# predictions = torch.tensor([1.0, 2.0, 3.0, 4.0])
# loss = variance_loss(predictions)
# print("Loss:", loss.item())

Losses = {
    'Sequence_CE': Sequence_CE,
    'Weighted_Gradient_MSE': Weighted_Gradient_MSE,
    'Variance_Weighted_MSE':Variance_Weighted_MSE
}

# -----------------------------------------------------------------------------#
# -------------------------------- lr_schedule --------------------------------#
# -----------------------------------------------------------------------------#


def get_lr_schedule_with_warmup(optimizer, num_training_steps,dataset,base_model,scale1=1/6, scale2=1/4,train=False,last_epoch=-1):
    
    if not train:
        if dataset == 'crosstask_how' and base_model == 'base':
            num_warmup_steps_scale = 1 / 6
            decay_steps_scale = 1 / 4
        elif dataset == 'crosstask_base' and base_model == 'base':
            num_warmup_steps_scale = 1 / 3
            decay_steps_scale = 1 / 2
        elif dataset == 'NIV' and base_model == 'base':
            num_warmup_steps_scale =  9 / 13
            decay_steps_scale =  3 / 13
        elif dataset == 'coin' and base_model == 'base':
            num_warmup_steps_scale = 1 / 40
            decay_steps_scale = 1 / 16
        elif dataset == 'crosstask_how' and base_model == 'predictor':
            num_warmup_steps_scale = 1 / 6
            decay_steps_scale = 1 / 4
        elif dataset == 'crosstask_base' and base_model == 'predictor':
            num_warmup_steps_scale = 1 / 3
            decay_steps_scale = 1 / 2
        elif dataset == 'NIV' and base_model == 'predictor':
            num_warmup_steps_scale =  1/5
            decay_steps_scale =  1/3
        elif dataset == 'coin' and base_model == 'predictor':
            num_warmup_steps_scale = 1 / 40
            decay_steps_scale = 1 / 16
        else:
            RuntimeError('select error!')
    else:
        num_warmup_steps_scale = scale1
        decay_steps_scale = scale2
        
    num_warmup_steps = int(num_training_steps * num_warmup_steps_scale)
    decay_steps = int(num_training_steps * decay_steps_scale)
    
    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return max(0., float(current_step) / float(max(1, num_warmup_steps)))
        else:
            return max(0.5 ** ((current_step - num_warmup_steps) // decay_steps), 0.)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -----------------------------------------------------------------------------#
# ---------------------------------- logging ----------------------------------#
# -----------------------------------------------------------------------------#

# Taken from PyTorch's examples.imagenet.main


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=SummaryWriter, if_exist=False):
        self._log_dir = log_dir
        print('logging outputs to ', log_dir)
        self._n_logged_samples = n_logged_samples
        self._summ_writer = summary_writer(
            log_dir, flush_secs=120, max_queue=10)
        if not if_exist:
            log = logging.getLogger(log_dir)
            if not log.handlers:
                log.setLevel(logging.DEBUG)
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter(
                    fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
                fh.setFormatter(formatter)
                log.addHandler(fh)
            self.log = log

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(
            group_name, phase), scalar_dict, step)

    def flush(self):
        self._summ_writer.flush()

    def log_info(self, info):
        self.log.info("{}".format(info))
