#!/usr/bin/env python3
"""
DDPM Scheduler for Diffusion Process
"""
import torch
import numpy as np

class DDPMScheduler:
    """DDPM noise scheduler for diffusion"""
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear'
    ):
        self.num_train_timesteps = num_train_timesteps
        
        # Create beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule from improved DDPM paper"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, original, noise, timesteps):
        """Add noise for forward diffusion"""
        alphas_cumprod = self.alphas_cumprod.to(original.device)
        
        # Reshape timesteps for broadcasting
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Expand dimensions to match original shape
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy
    
    def step(self, model_output, timestep, sample):
        """Single denoising step"""
        t = timestep.item() if torch.is_tensor(timestep) else timestep
        
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        beta_prod_t = 1 - alpha_prod_t
        
        # Predicted original sample
        pred_original = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # Compute previous sample
        if t > 0:
            noise = torch.randn_like(sample)
            alpha_prod_t_prev = self.alphas_cumprod[t - 1]
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
            prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original
            prev_sample = prev_sample + (variance ** 0.5) * noise
        else:
            prev_sample = pred_original
        
        return prev_sample
    
    def set_timesteps(self, num_inference_steps):
        """Set timesteps for inference"""
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        ).long()
