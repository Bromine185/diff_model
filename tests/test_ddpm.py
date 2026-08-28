"""Diffusion arithmetic. These catch the failures that are otherwise silent.

A DDPM with an off-by-one in alphabar still trains, still shows a falling loss
curve, and still produces images — just wrong ones. There is no runtime error to
notice, so the arithmetic has to be pinned by tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from diffmodel.config import SMOKE
from diffmodel.ddpm import (
    DiffusionSchedule,
    EMA,
    cosine_beta_schedule,
    linear_beta_schedule,
    p_losses,
    q_sample,
    sample,
)
from diffmodel.unet import UNet


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


# --- schedules ------------------------------------------------------------

@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_buffers_are_consistent(kind):
    sched = DiffusionSchedule.build(1000, kind=kind)
    assert len(sched) == 1000
    assert torch.all(sched.betas > 0) and torch.all(sched.betas < 1)
    # alphabar is a cumulative product of alphas
    assert torch.allclose(sched.alphas_cumprod, torch.cumprod(sched.alphas, 0), atol=1e-5)
    # alphabar decreases monotonically to ~0
    assert torch.all(sched.alphas_cumprod[1:] <= sched.alphas_cumprod[:-1] + 1e-7)
    assert sched.alphas_cumprod[-1] < 1e-3, "chain must reach ~pure noise"
    # the shifted buffer really is shifted
    assert sched.alphas_cumprod_prev[0] == pytest.approx(1.0)
    assert torch.allclose(sched.alphas_cumprod_prev[1:], sched.alphas_cumprod[:-1], atol=1e-6)
    # final reverse step is deterministic
    assert sched.posterior_variance[0] == pytest.approx(0.0, abs=1e-9)


def test_short_schedule_warns_that_it_never_reaches_noise():
    """T=50 with the default endpoints leaves alphabar_T far from 0.

    Exactly the trap a SMOKE preset falls into. It must warn, not sail past.
    """
    with pytest.warns(UserWarning, match="does not reach pure noise"):
        DiffusionSchedule.build(50, kind="linear")


def test_cosine_reaches_noise_at_small_T():
    """The cosine schedule is the fix for small T, so it must not warn."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sched = DiffusionSchedule.build(50, kind="cosine")
    assert sched.alphas_cumprod[-1] < 1e-3


def test_linear_schedule_endpoints():
    b = linear_beta_schedule(1000, 1e-4, 0.02)
    assert b[0] == pytest.approx(1e-4)
    assert b[-1] == pytest.approx(0.02)


def test_cosine_schedule_is_bounded():
    b = cosine_beta_schedule(1000)
    assert torch.all(b > 0) and torch.all(b < 1)


# --- the forward process --------------------------------------------------

def test_forward_moments_match_closed_form():
    """THE test. Iterate q(x_t | x_{t-1}) explicitly and compare to q(x_t | x_0).

    The closed form is what training relies on; the iterated chain is the
    definition. If they disagree, alphabar is misindexed and every sample the
    model ever produces is wrong. Checked at several t across the chain.
    """
    T = 200
    sched = DiffusionSchedule.build(T, kind="cosine")
    n = 20000
    x0 = torch.full((n, 1), 2.0)

    for t_target in (0, 1, 25, 99, T - 1):
        # iterate the definition: x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) z
        x = x0.clone()
        for t in range(t_target + 1):
            beta = sched.betas[t]
            x = torch.sqrt(1.0 - beta) * x + torch.sqrt(beta) * torch.randn_like(x)
        emp_mean, emp_std = x.mean().item(), x.std().item()

        # closed form
        exp_mean = (sched.sqrt_alphas_cumprod[t_target] * 2.0).item()
        exp_std = sched.sqrt_one_minus_alphas_cumprod[t_target].item()

        assert emp_mean == pytest.approx(exp_mean, abs=0.05), f"mean mismatch at t={t_target}"
        assert emp_std == pytest.approx(exp_std, abs=0.05), f"std mismatch at t={t_target}"


def test_q_sample_is_exact_given_the_noise():
    """With eps supplied, q_sample is a deterministic affine map — check it directly."""
    sched = DiffusionSchedule.build(100, kind="cosine")
    x0 = torch.randn(8, 3, 8, 128)
    eps = torch.randn_like(x0)
    t = torch.full((8,), 42, dtype=torch.long)
    xt, returned = q_sample(x0, t, sched, noise=eps)
    a = sched.sqrt_alphas_cumprod[42]
    b = sched.sqrt_one_minus_alphas_cumprod[42]
    assert torch.allclose(xt, a * x0 + b * eps, atol=1e-6)
    assert torch.equal(returned, eps)


def test_q_sample_at_t0_barely_perturbs():
    sched = DiffusionSchedule.build(1000, kind="linear")
    x0 = torch.randn(64, 3, 8, 128)
    t = torch.zeros(64, dtype=torch.long)
    xt, _ = q_sample(x0, t, sched)
    assert (xt - x0).abs().mean() < 0.02


def test_q_sample_at_final_t_is_pure_noise():
    sched = DiffusionSchedule.build(1000, kind="linear")
    x0 = torch.randn(2000, 1) * 5.0 + 10.0   # far from N(0,1)
    t = torch.full((2000,), 999, dtype=torch.long)
    xt, _ = q_sample(x0, t, sched)
    assert abs(xt.mean().item()) < 0.15
    assert xt.std().item() == pytest.approx(1.0, abs=0.1)


def test_gather_broadcasts_per_sample():
    """Different t per element must produce different scalings — a real bug source."""
    sched = DiffusionSchedule.build(1000, kind="linear")
    x0 = torch.ones(3, 1, 1, 1)
    t = torch.tensor([0, 500, 999])
    xt, _ = q_sample(x0, t, sched, noise=torch.zeros_like(x0))
    vals = xt.flatten()
    assert vals[0] > vals[1] > vals[2], "larger t must shrink the signal more"


# --- training / sampling --------------------------------------------------

def _tiny_model():
    return UNet(in_channels=3, base_channels=8, channel_mults=(1, 2),
                num_res_blocks=1, attention_at_bottleneck=False)


def test_p_losses_is_finite_and_differentiable():
    sched = DiffusionSchedule.build(100, kind="cosine")
    model = _tiny_model()
    x0 = torch.randn(4, 3, 8, 128)
    t = torch.randint(0, 100, (4,))
    loss = p_losses(model, x0, t, sched)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_zero_init_gives_unit_initial_loss():
    """The final conv is zero-initialised, so eps_theta = 0 and the loss is E[eps^2] = 1."""
    sched = DiffusionSchedule.build(100, kind="cosine")
    model = _tiny_model()
    x0 = torch.randn(256, 3, 8, 128)
    t = torch.randint(0, 100, (256,))
    assert p_losses(model, x0, t, sched).item() == pytest.approx(1.0, abs=0.06)


def test_sample_returns_correct_shape_and_is_finite():
    sched = DiffusionSchedule.build(20, kind="cosine")
    model = _tiny_model()
    out = sample(model, (2, 3, 8, 128), sched, torch.device("cpu"))
    assert out.shape == (2, 3, 8, 128)
    assert torch.isfinite(out).all()


def test_sample_trajectory_is_ordered_and_ends_at_result():
    sched = DiffusionSchedule.build(20, kind="cosine")
    model = _tiny_model()
    out, traj = sample(model, (1, 3, 8, 128), sched, torch.device("cpu"), return_trajectory=True)
    assert len(traj) >= 2
    assert torch.allclose(traj[-1], out.cpu(), atol=1e-6), "last trajectory frame is the result"


def test_sampling_is_reproducible_under_a_fixed_seed():
    sched = DiffusionSchedule.build(20, kind="cosine")
    model = _tiny_model()
    torch.manual_seed(123)
    a = sample(model, (2, 3, 8, 128), sched, torch.device("cpu"))
    torch.manual_seed(123)
    b = sample(model, (2, 3, 8, 128), sched, torch.device("cpu"))
    assert torch.equal(a, b)


# --- EMA ------------------------------------------------------------------

def test_ema_tracks_but_lags_the_live_weights():
    model = _tiny_model()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    live = torch.cat([p.flatten() for p in model.parameters()])
    shadow = torch.cat([p.flatten() for p in ema.shadow.parameters()])
    assert not torch.allclose(live, shadow), "EMA must lag"
    # after one update at decay 0.9 the shadow moved 10% of the way
    assert torch.allclose(shadow, live - 0.9, atol=1e-5)


def test_ema_converges_to_live_weights_after_many_updates():
    model = _tiny_model()
    ema = EMA(model, decay=0.5)
    for _ in range(60):
        ema.update(model)
    live = torch.cat([p.flatten() for p in model.parameters()])
    shadow = torch.cat([p.flatten() for p in ema.shadow.parameters()])
    assert torch.allclose(live, shadow, atol=1e-6)


# --- config coherence -----------------------------------------------------

def test_smoke_preset_builds_a_working_stack():
    """The SMOKE preset must actually run, since it is the local correctness gate."""
    cfg = SMOKE
    sched = DiffusionSchedule.build(cfg.timesteps, kind="cosine")
    model = UNet(3, cfg.base_channels, cfg.channel_mults, cfg.num_res_blocks,
                 cfg.attention_at_bottleneck)
    x0 = torch.randn(cfg.batch_size, *cfg.image_shape)
    t = torch.randint(0, cfg.timesteps, (cfg.batch_size,))
    assert torch.isfinite(p_losses(model, x0, t, sched))
