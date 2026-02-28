# -*- coding: utf-8 -*-
"""
Diffusion model for action trajectory generation.
Conditioned on state embedding; outputs continuous action sequence (H, robot_num, 2).
Used to replace tree search: sample N trajectories, rollout with state_predictor + value_estimator, pick best.

Training: set config.model_predictive_rl.use_diffusion = True and train diffusion with BC:
  - Run policy with use_diffusion=False (tree search) to collect (state, action_sequence).
  - action_sequence = next H actions from policy. Train diffusion with forward() loss (noise prediction).
  - Then set use_diffusion=True and load diffusion checkpoint; or train diffusion jointly (see trainer).
"""
import math
import torch
import torch.nn as nn
from configs.config import BaseEnvConfig


def _beta_schedule(num_steps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_steps)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ActionTrajectoryDiffusion(nn.Module):
    """
    DDPM over flattened action trajectory. Condition: state embedding.
    Input: state = (robot_states, human_states) tensors with batch; graph_model to get embedding.
    Output: action sequence (batch, horizon, robot_num, 2) in [-1, 1], scale to env with action_scale.
    """

    def __init__(self, state_embed_dim, horizon, robot_num, num_steps=100,
                 beta_start=1e-4, beta_end=0.02, hidden_dim=256):
        super().__init__()
        self.horizon = horizon
        self.robot_num = robot_num
        self.num_steps = num_steps
        self.action_dim = horizon * robot_num * 2  # flattened
        self.state_embed_dim = state_embed_dim
        self.action_scale = 30.0  # one_uav max |dx|,|dy|

        betas = _beta_schedule(num_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        time_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noise_pred_net = nn.Sequential(
            nn.Linear(self.action_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, self.action_dim),
        )

    def get_state_embedding(self, state, graph_model):
        """state = (robot_states, human_states) batch tensors; return (batch, state_embed_dim)."""
        tmp_config = BaseEnvConfig()
        robot_num = tmp_config.env.robot_num
        out = graph_model(state)
        state_embed = out[:, 0:robot_num, :].reshape(out.size(0), -1)
        return state_embed

    def _scale_to_env(self, x):
        """x in [-1, 1] -> env action scale."""
        return x * self.action_scale

    def _scale_to_norm(self, a):
        """env action -> [-1, 1]."""
        return a / self.action_scale

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0, device=x_0.device)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1).to(x_0.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1).to(x_0.device)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def predict_noise(self, x_t, t, state_embed):
        t_emb = self.time_mlp(t)
        s_emb = self.state_mlp(state_embed)
        h = torch.cat([x_t, t_emb, s_emb], dim=-1)
        return self.noise_pred_net(h)

    def forward(self, state, graph_model, action_trajectory):
        """
        Training: predict noise given noisy trajectory and state.
        action_trajectory: (batch, horizon, robot_num, 2) normalized in [-1,1].
        """
        batch = action_trajectory.size(0)
        state_embed = self.get_state_embedding(state, graph_model)
        x_0 = action_trajectory.reshape(batch, -1)
        t = torch.randint(0, self.num_steps, (batch,), device=action_trajectory.device).long()
        x_t, noise = self.add_noise(x_0, t)
        pred_noise = self.predict_noise(x_t, t, state_embed)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, state, graph_model, num_samples=1):
        """
        Sample action trajectories. state = (robot_states, human_states) with batch=1 typically.
        Returns (num_samples, horizon, robot_num, 2) in [-1, 1].
        """
        device = next(self.parameters()).device
        state_embed = self.get_state_embedding(state, graph_model)
        if state_embed.size(0) == 1 and num_samples > 1:
            state_embed = state_embed.repeat(num_samples, 1)
        elif state_embed.size(0) != num_samples:
            state_embed = state_embed[:num_samples]
        batch = state_embed.size(0)

        x = torch.randn(batch, self.action_dim, device=device)
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
            pred_noise = self.predict_noise(x, t_batch, state_embed)
            alpha = self.alphas_cumprod[t].to(device)
            alpha_prev = self.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1.0, device=device)
            beta = 1 - alpha / alpha_prev
            # DDPM: x_{t-1} = (x_t - (beta/sqrt(1-alpha))*pred_noise)/sqrt(alpha) + sqrt(beta)*z
            x = (x - (beta / torch.sqrt(1 - alpha)) * pred_noise) / torch.sqrt(alpha)
            if t > 0:
                noise = torch.randn_like(x, device=device)
                x = x + torch.sqrt(beta) * noise
        x = torch.clamp(x, -1.0, 1.0)
        return x.reshape(batch, self.horizon, self.robot_num, 2)
