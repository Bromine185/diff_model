"""The denoising diffusion probabilistic model, derived and implemented.

Ho, Jain & Abbeel (2020). Written out rather than imported so the arithmetic is
inspectable, because almost every silent failure in a DDPM is an indexing error
in these buffers rather than a bug in the network.

The forward process
-------------------
Corrupt data by adding Gaussian noise in T small steps, with a fixed variance
schedule beta_1 .. beta_T:

    q(x_t | x_{t-1}) = N( sqrt(1 - beta_t) * x_{t-1},  beta_t * I )

Writing alpha_t = 1 - beta_t and alphabar_t = prod_{s<=t} alpha_s, the Gaussians
compose in closed form, so *any* noise level is one step away from clean data:

    q(x_t | x_0) = N( sqrt(alphabar_t) * x_0,  (1 - alphabar_t) * I )

    => x_t = sqrt(alphabar_t) * x_0 + sqrt(1 - alphabar_t) * eps,  eps ~ N(0, I)

That identity is why training is cheap: sample a random t per example rather than
simulating the chain. `test_forward_moments_match_closed_form` checks the closed
form against an explicitly iterated chain, which is the test that catches an
off-by-one in alphabar.

The reverse process
-------------------
The true posterior q(x_{t-1} | x_t, x_0) is Gaussian with

    mu~(x_t, x_0) = ( sqrt(alphabar_{t-1}) * beta_t / (1 - alphabar_t) ) * x_0
                  + ( sqrt(alpha_t) * (1 - alphabar_{t-1}) / (1 - alphabar_t) ) * x_t
    beta~_t       = (1 - alphabar_{t-1}) / (1 - alphabar_t) * beta_t

We do not know x_0 at sampling time, but the forward identity can be rearranged:

    x_0 = ( x_t - sqrt(1 - alphabar_t) * eps ) / sqrt(alphabar_t)

so predicting eps is equivalent to predicting x_0, and substituting gives the
mean the sampler actually uses:

    mu_theta(x_t, t) = 1/sqrt(alpha_t) * ( x_t - beta_t/sqrt(1 - alphabar_t) * eps_theta(x_t, t) )

Ho et al. showed the variational bound reduces, up to a weighting the authors drop,
to plain MSE between the true and predicted noise. That is the whole objective.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------
# Noise schedules
# --------------------------------------------------------------------------

def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 0.02) -> torch.Tensor:
    """The original DDPM schedule, and the one the paper follows via the HF tutorial.

    Tuned for T = 1000. At smaller T the endpoints must move or the chain will not
    reach pure noise — `DiffusionSchedule` checks that and warns rather than
    letting a SMOKE run silently train against a chain that never fully corrupts.
    """
    return torch.linspace(start, end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021).

    The linear schedule destroys information too quickly at the end of the chain,
    so the last several hundred steps carry little signal. The cosine schedule
    defines alphabar directly and spreads the corruption more evenly. Offered as a
    switch; the paper uses linear, so linear is the default.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


@dataclass
class DiffusionSchedule:
    """Every buffer the forward and reverse processes need, precomputed once.

    Stored in float64 during construction and cast to float32 at the end:
    alphabar is a cumulative product over up to 1000 terms and accumulates real
    error in float32, which shows up as a sampler that never quite converges.
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor

    @classmethod
    def build(cls, timesteps: int, kind: str = "linear", start: float = 1e-4, end: float = 0.02):
        if kind == "linear":
            betas = linear_beta_schedule(timesteps, start, end)
        elif kind == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {kind!r}")

        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1, dtype=torch.float64), acp[:-1]])

        # If alphabar_T is not close to 0 the chain never reaches pure noise, and
        # sampling from N(0, I) starts off-distribution. Easy to hit by shrinking T
        # without moving the schedule endpoints.
        if acp[-1] > 1e-3:
            import warnings
            warnings.warn(
                f"alphabar_T = {acp[-1]:.4g} is not ~0 with T={timesteps} and the "
                f"'{kind}' schedule: the forward chain does not reach pure noise, so "
                f"samples will start off-distribution. Raise beta_end or use "
                f"kind='cosine' for small T.",
                stacklevel=2,
            )

        f32 = lambda t: t.to(torch.float32)  # noqa: E731
        return cls(
            betas=f32(betas),
            alphas=f32(alphas),
            alphas_cumprod=f32(acp),
            alphas_cumprod_prev=f32(acp_prev),
            sqrt_alphas_cumprod=f32(torch.sqrt(acp)),
            sqrt_one_minus_alphas_cumprod=f32(torch.sqrt(1.0 - acp)),
            sqrt_recip_alphas=f32(torch.sqrt(1.0 / alphas)),
            # beta~_t. Element 0 is 0 by construction (alphabar_{-1} = 1), which is
            # correct: the final reverse step is deterministic.
            posterior_variance=f32(betas * (1.0 - acp_prev) / (1.0 - acp)),
        )

    def to(self, device: torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(**{k: v.to(device) for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        return len(self.betas)


def _gather(buf: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Index a per-timestep buffer and reshape to broadcast over (B, C, H, W)."""
    out = buf.gather(0, t)
    return out.reshape(-1, *([1] * (ndim - 1)))


# --------------------------------------------------------------------------
# Forward / reverse
# --------------------------------------------------------------------------

def q_sample(
    x0: torch.Tensor, t: torch.Tensor, sched: DiffusionSchedule, noise: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jump straight to noise level t: x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps."""
    if noise is None:
        noise = torch.randn_like(x0)
    a = _gather(sched.sqrt_alphas_cumprod, t, x0.ndim)
    b = _gather(sched.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
    return a * x0 + b * noise, noise


def p_losses(
    model: nn.Module,
    x0: torch.Tensor,
    t: torch.Tensor,
    sched: DiffusionSchedule,
    noise: torch.Tensor | None = None,
    loss_type: str = "l2",
) -> torch.Tensor:
    """The entire training objective: MSE between true and predicted noise."""
    x_t, noise = q_sample(x0, t, sched, noise)
    pred = model(x_t, t)
    if loss_type == "l2":
        return F.mse_loss(pred, noise)
    if loss_type == "l1":
        return F.l1_loss(pred, noise)
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, noise)
    raise ValueError(f"unknown loss_type {loss_type!r}")


@torch.no_grad()
def p_sample_step(
    model: nn.Module, x: torch.Tensor, t_index: int, sched: DiffusionSchedule
) -> torch.Tensor:
    """One reverse step, x_t -> x_{t-1}."""
    t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)
    beta = _gather(sched.betas, t, x.ndim)
    sqrt_omab = _gather(sched.sqrt_one_minus_alphas_cumprod, t, x.ndim)
    sqrt_recip_a = _gather(sched.sqrt_recip_alphas, t, x.ndim)

    mean = sqrt_recip_a * (x - beta / sqrt_omab * model(x, t))
    if t_index == 0:
        return mean  # final step is deterministic; beta~_0 = 0
    var = _gather(sched.posterior_variance, t, x.ndim)
    return mean + torch.sqrt(var) * torch.randn_like(x)


@torch.no_grad()
def sample(
    model: nn.Module,
    shape: tuple[int, ...],
    sched: DiffusionSchedule,
    device: torch.device,
    return_trajectory: bool = False,
    progress: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the full reverse chain from pure noise.

    `return_trajectory` keeps ~12 evenly spaced intermediates for the denoising
    strip in the notebook — the picture that shows noise resolving into wavelet
    structure.
    """
    model.eval()
    x = torch.randn(shape, device=device)
    T = len(sched)
    keep = set(range(T - 1, -1, -max(1, T // 12))) | {0}
    traj: list[torch.Tensor] = []

    steps = range(T - 1, -1, -1)
    if progress:
        from tqdm.auto import tqdm

        steps = tqdm(steps, desc="sampling", total=T, leave=False)

    for i in steps:
        x = p_sample_step(model, x, i, sched)
        if return_trajectory and i in keep:
            traj.append(x.detach().cpu().clone())

    return (x, traj) if return_trajectory else x


# --------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------

class EMA:
    """Exponential moving average of weights, sampled from instead of the live ones.

    Standard for diffusion models and not optional in practice: the raw SGD
    iterate bounces around a broad optimum and produces visibly worse samples than
    its own running average. Keeps a full shadow copy, so it costs one extra model
    in memory.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd) -> None:
        self.shadow.load_state_dict(sd)
