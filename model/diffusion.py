import random
import numpy as np
import torch
from torch import nn

from .helpers import (
    cosine_beta_schedule,
    extract,
    condition_projection,
    Losses,
)


class GaussianDiffusion(nn.Module):
    def __init__(self, args, model, ddim_discr_method='uniform'):
        super().__init__()
        self.horizon = args.horizon  # Set the horizon (sequence length)
        self.observation_dim = args.observation_dim  # Set the observation dimension
        self.action_dim = args.action_dim  # Set the action dimension
        self.class_dim = args.class_dim  # Set the class dimension
        self.model = model  # Set the model (e.g., UNet)
        self.weight = args.weight  # Set the weight for the loss function
        self.ifMask = args.ifMask

        # Set the number of timesteps for diffusion
        self.n_timesteps = args.n_diffusion_steps # default=200
        self.clip_denoised = args.clip_denoised  # Whether to clip the denoised output, default=True
        self.eta = 0.0  # Eta parameter for DDIM sampling
        self.random_ratio = 1.0  # Ratio of random noise
        
        self.l_order=args.l_order
        self.l_pos=args.l_pos
        self.l_perm=args.l_perm
        
        # Calculate the beta schedule using a cosine function
        betas = cosine_beta_schedule(self.n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # 计算dim=0上的累积乘积
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # ---------------------------ddim (Denoising Diffusion Implicit Models)--------------------------------
        ddim_timesteps = 10

        if ddim_discr_method == 'uniform':
            c = self.n_timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.n_timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(self.n_timesteps), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise RuntimeError("Unknown DDIM discretization method")

        self.ddim_timesteps = ddim_timesteps
        self.ddim_timestep_seq = ddim_timestep_seq
        # ----------------------------------------------------------------

        # Register buffers for various terms used in the diffusion process
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # Set the loss type and corresponding loss function
        self.loss_type = args.loss_type
        self.loss_fn = Losses[self.loss_type](
            self.action_dim, self.class_dim, self.weight)

    # ------------------------------------------ sampling ------------------------------------------#

    # Function to compute the posterior mean and variance
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # Function to compute the mean and variance of the model's predictions
    def p_mean_variance(self, x, cond, t, mask=None):
        x_recon = self.model(x, t)  # Reconstruct x from the model
        
        if mask is not None:
            x_recon = x_recon * mask
            
        if self.clip_denoised:
            x_recon.clamp(-1., 1.)  # Clip the denoised output if specified
        else:
            raise RuntimeError("Clipping denoised output is disabled")

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # Function to predict epsilon from x_t and x_start
    @torch.no_grad()
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return \
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) \
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # Function to perform DDIM sampling
    @torch.no_grad()
    def p_sample_ddim(self, x, cond, t, t_prev, if_prev=False):
        b, *_, device = *x.shape, x.device
        x_recon = self.model(x, t)  # Reconstruct x from the model

        if self.clip_denoised:
            x_recon.clamp(-1., 1.)  # Clip the denoised output if specified
        else:
            raise RuntimeError("Clipping denoised output is disabled")

        eps = self._predict_eps_from_xstart(x, t, x_recon)
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        if if_prev:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x) * self.random_ratio
        mean_pred = (
            x_recon * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return mean_pred + nonzero_mask * sigma * noise

    # Function to perform standard diffusion model sampling
    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        # t是当前的时间步，一个标量或一维张量，指示我们处于去噪过程中的哪个阶段
        # model_mean 模型预测的均值
        # 模型预测的对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t) 
        noise = torch.randn_like(x) * self.random_ratio
        # (t == 0): 返回一个布尔张量，其中 ：
        # t 为 0 的位置为 True，否则为 False。
        # (1 - (t == 0).float()): 将布尔张量转换为浮点张量，并将 True 变为 0.0，False 变为 1.0。
        # reshape(b, *((1,) * (len(x.shape) - 1))): 重塑为与 x 形状匹配的张量，其中第一个维度为 b，其他维度为 1。
        # nonzero_mask: 创建一个掩码张量，其中 t 非 0 的位置为 1.0，其他位置为 0.0。
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # Loop for sampling, supporting both diffusion and DDIM methods
    @torch.no_grad()
    def p_sample_loop(self, cond, if_jump):
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.class_dim +
                 self.action_dim + self.observation_dim)

        x = torch.randn(shape, device=device) * \
            self.random_ratio  # Initialize xt for Noise and diffusion
        # x = torch.zeros(shape, device=device)   # for Deterministic
        x = condition_projection(x, cond, self.action_dim, self.class_dim)

        '''
        The if-else below is for diffusion, should be removed for Noise and Deterministic
        '''
        if not if_jump:
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full(
                    (batch_size,), i, device=device, dtype=torch.long) # 用于创建一个给定形状的张量，并填充指定的值
                x = self.p_sample(x, cond, timesteps)
                x = condition_projection(
                    x, cond, self.action_dim, self.class_dim)

        else:
            for i in reversed(range(0, self.ddim_timesteps)):
                timesteps = torch.full(
                    (batch_size,), self.ddim_timestep_seq[i], device=device, dtype=torch.long)
                if i == 0:
                    timesteps_prev = torch.full(
                        (batch_size,), 0, device=device, dtype=torch.long)
                    x = self.p_sample_ddim(
                        x, cond, timesteps, timesteps_prev, True)
                else:
                    timesteps_prev = torch.full(
                        (batch_size,), self.ddim_timestep_seq[i-1], device=device, dtype=torch.long)
                    x = self.p_sample_ddim(x, cond, timesteps, timesteps_prev)
                x = condition_projection(
                    x, cond, self.action_dim, self.class_dim)

        '''
        The two lines below are for Noise and Deterministic
        '''
        # x = self.model(x, None)
        # x = condition_projection(x, cond, self.action_dim, self.class_dim)

        return x

    # ------------------------------------------ training ------------------------------------------#

    # Function to sample from q(x_t | x_0) using the forward diffusion process
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start) * self.random_ratio

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

        return sample

    # Compute the loss for the denoising process
    def p_losses(self, x_start, cond, t):

        # print("p_loss")
        # Generate noise for Noise and diffusion
        noise = torch.randn_like(x_start) * self.random_ratio
        # noise = torch.zeros_like(x_start)   # for Deterministic
        # x_noisy = noise   # for Noise and Deterministic

        # For diffusion, add noise to the input
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_noisy = condition_projection(
            x_noisy, cond, self.action_dim, self.class_dim)

        x_recon = self.model(x_noisy, t)  # Reconstruct from noisy input

        # x_recon = x_noisy - x_recon

        x_recon = condition_projection(
            x_recon, cond, self.action_dim, self.class_dim)
        
        
        if self.loss_type == 'Sequence_CE':        
             loss = self.loss_fn(x_recon, x_start, self.l_order,self.l_pos,self.l_perm)  # Compute the loss
        else:
            loss = self.loss_fn(x_recon, x_start)  # Compute the loss
        return loss

    # Compute the loss for the batch
    def loss(self, x, cond):
        batch_size = len(x)  # Get the batch size

        # t.shape = (batch_size,)
        t = torch.randint(0, self.n_timesteps, (batch_size,),
                          device=x.device).long()  # Random timestep for diffusion
        # t = None    # for Noise and Deterministic
        return self.p_losses(x, cond, t)

    # Forward pass of the model
    def forward(self, cond, if_jump=False):
        return self.p_sample_loop(cond, if_jump)
