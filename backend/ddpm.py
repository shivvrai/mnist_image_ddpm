import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SZ = 28
CH = 1
CLS = 10
NULL_CLS = CLS
TIMESTEPS = 300

# ---------------------------------------------------------------------------
# Noise schedule (precomputed constants — same math as the TF version)
# ---------------------------------------------------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    x = torch.arange(timesteps + 1, dtype=torch.float32)
    ac = torch.cos(((x / (timesteps + 1)) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 1e-8, 0.999)

_betas = cosine_beta_schedule(TIMESTEPS).numpy()
_alphas = 1.0 - _betas
_alpha_hats = np.cumprod(_alphas)

# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------
def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / float(half)
    )
    t = t.float().unsqueeze(1)
    x = t * freqs.unsqueeze(0)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, channels, cond_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.t_proj = nn.Linear(cond_dim, channels)

    def forward(self, x, cond_emb):
        h = F.silu(x)
        h = self.conv1(h)
        t = self.t_proj(cond_emb).unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        h = h + t
        h = F.silu(h)
        h = self.conv2(h)
        return x + h

# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, base_filters=64, cond_dim=128):
        super().__init__()
        bf = base_filters

        # --- conditioning ---
        self.class_embed = nn.Embedding(CLS + 1, cond_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )

        # --- encoder ---
        self.init_conv = nn.Conv2d(CH, bf, 3, padding=1)
        self.enc1_res1 = ResBlock(bf, cond_dim)
        self.enc1_res2 = ResBlock(bf, cond_dim)

        self.down1 = nn.Conv2d(bf, bf * 2, 3, stride=2, padding=0)
        self.enc2_res1 = ResBlock(bf * 2, cond_dim)
        self.enc2_res2 = ResBlock(bf * 2, cond_dim)

        self.down2 = nn.Conv2d(bf * 2, bf * 4, 3, stride=2, padding=0)
        self.bottle_res1 = ResBlock(bf * 4, cond_dim)
        self.bottle_res2 = ResBlock(bf * 4, cond_dim)

        # --- decoder ---
        dec1_ch = bf * 4 + bf * 2              # 256 + 128 = 384
        self.dec1_res1 = ResBlock(dec1_ch, cond_dim)
        self.dec1_res2 = ResBlock(dec1_ch, cond_dim)

        dec2_ch = dec1_ch + bf                  # 384 + 64 = 448
        self.dec2_res1 = ResBlock(dec2_ch, cond_dim)
        self.dec2_res2 = ResBlock(dec2_ch, cond_dim)

        self.final_conv = nn.Conv2d(dec2_ch, CH, 3, padding=1)

    def forward(self, img, t, y):
        # conditioning
        t_emb = timestep_embedding(t, 128)
        y_emb = self.class_embed(y)
        cond = self.cond_mlp(t_emb + y_emb)

        # encoder
        x = self.init_conv(img)
        x = self.enc1_res1(x, cond)
        s1 = self.enc1_res2(x, cond)

        x = F.pad(s1, (0, 1, 0, 1))
        x = self.down1(x)
        x = self.enc2_res1(x, cond)
        s2 = self.enc2_res2(x, cond)

        x = F.pad(s2, (0, 1, 0, 1))
        x = self.down2(x)
        x = self.bottle_res1(x, cond)
        x = self.bottle_res2(x, cond)

        # decoder
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, s2], dim=1)
        x = self.dec1_res1(x, cond)
        x = self.dec1_res2(x, cond)

        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, s1], dim=1)
        x = self.dec2_res1(x, cond)
        x = self.dec2_res2(x, cond)

        return self.final_conv(x)

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------
def _p_sample_step(x, eps, beta, alpha, alpha_cum):
    coef1 = 1.0 / math.sqrt(alpha)
    coef2 = (1.0 - alpha) / math.sqrt(1.0 - alpha_cum)
    mean = coef1 * (x - coef2 * eps)
    sigma = math.sqrt(beta)
    return mean + sigma * torch.randn_like(x)


@torch.no_grad()
def gen_samples(model, n_samples, conditioning, guidance_scale, seed=None):
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn(n_samples, CH, IMG_SZ, IMG_SZ, device=device)
    cond = torch.tensor(conditioning, dtype=torch.long, device=device)
    null_cond = torch.full((n_samples,), NULL_CLS, dtype=torch.long, device=device)

    for step in reversed(range(TIMESTEPS)):
        t_batch = torch.full((n_samples,), step, dtype=torch.long, device=device)
        eps_u = model(x, t_batch, null_cond)
        eps_c = model(x, t_batch, cond)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = _p_sample_step(x, eps, _betas[step], _alphas[step], _alpha_hats[step])

    return (x + 1.0) * 0.5          # → [0, 1]


@torch.no_grad()
def gen_samples_from_image(model, x_start, strength, conditioning, guidance_scale, seed=None):
    """Image-to-image generation: add noise to a sketch, then denoise."""
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    x_start = x_start.to(device).float()
    n_samples = x_start.shape[0]
    t_start = int(strength * TIMESTEPS)

    if t_start == 0:
        return (x_start + 1.0) * 0.5

    noise = torch.randn_like(x_start)
    alpha_hat_t = _alpha_hats[t_start - 1]
    x = math.sqrt(alpha_hat_t) * x_start + math.sqrt(1 - alpha_hat_t) * noise

    cond = torch.tensor(conditioning, dtype=torch.long, device=device)
    null_cond = torch.full((n_samples,), NULL_CLS, dtype=torch.long, device=device)

    for step in reversed(range(t_start)):
        t_batch = torch.full((n_samples,), step, dtype=torch.long, device=device)
        eps_u = model(x, t_batch, null_cond)
        eps_c = model(x, t_batch, cond)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
        x = _p_sample_step(x, eps, _betas[step], _alphas[step], _alpha_hats[step])

    return (x + 1.0) * 0.5
