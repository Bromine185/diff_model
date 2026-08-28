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
    p_sample_step,
    q_sample,
    sample,
)
from diffmodel.seeding import torch_generator
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
    # posterior-mean coefficients: pushing the ZERO-NOISE forward state
    # x_t = sqrt(alphabar_t) x0 through the posterior mean must land exactly on
    # the zero-noise x_{t-1}, i.e. coef1 + coef2 * sqrt(alphabar_t) =
    # sqrt(alphabar_{t-1}) — an identity that fails if either coefficient is
    # mis-derived from Ho et al. eq. 7.
    lhs = sched.posterior_mean_coef1 + sched.posterior_mean_coef2 * sched.sqrt_alphas_cumprod
    assert torch.allclose(lhs, torch.sqrt(sched.alphas_cumprod_prev), atol=1e-5)


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
    """Upper clip is 0.999, per Nichol & Dhariwal §3.2 — NOT 0.9999.

    The distinction is load-bearing at small T: the clip caps the reverse step's
    state gain 1/sqrt(alpha_t) at sqrt(1/0.001) ≈ 31.6. The earlier 0.9999 clip
    allowed alpha_t = 1e-4, i.e. a 100x gain on the FIRST reverse step, which is
    what let a merely-inaccurate epsilon predictor blow SMOKE samples up to
    std ~1.9e3 (RESEARCH.md Phase 4 correction).
    """
    for T in (50, 1000):
        b = cosine_beta_schedule(T)
        assert torch.all(b > 0) and torch.all(b <= 0.999)
    # at T=50 the schedule actually saturates the clip, so the bound is exercised
    assert cosine_beta_schedule(50).max() == pytest.approx(0.999)


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


# --- the x0-parameterised reverse step ------------------------------------

def _eps_form_step(model, x, t_index, sched, noise):
    """Reference implementation: the epsilon-parameterised reverse step.

    mu = 1/sqrt(alpha_t) * (x - beta_t / sqrt(1 - alphabar_t) * eps_theta), the
    exact update `p_sample_step` ran before it was rewritten in x0-hat form.
    Kept here as the independent reference the x0-hat form must reproduce: the
    posterior-mean coefficients c1 = beta_t sqrt(alphabar_{t-1}) / (1 - alphabar_t)
    and c2 = (1 - alphabar_{t-1}) sqrt(alpha_t) / (1 - alphabar_t) (Ho et al.
    2020, eq. 7) collapse to this line when x0_hat is substituted unclamped.
    """
    t = torch.full((x.shape[0],), t_index, dtype=torch.long)
    beta = sched.betas[t_index]
    sqrt_omab = sched.sqrt_one_minus_alphas_cumprod[t_index]
    sqrt_recip_a = sched.sqrt_recip_alphas[t_index]
    with torch.no_grad():
        mean = sqrt_recip_a * (x - beta / sqrt_omab * model(x, t))
    if t_index == 0:
        return mean
    return mean + torch.sqrt(sched.posterior_variance[t_index]) * noise


def test_x0_form_step_equals_eps_form_when_unclamped():
    """x0_clamp=None makes the rewritten step ALGEBRAICALLY the old one.

    Checked at both ends of the chain and in the middle, on the SMOKE schedule
    (T=50 cosine) where the arithmetic is at its most ill-conditioned — the
    t=T-1 step divides by sqrt(alphabar_T) ~ 1e-3, so this is also where an
    algebra slip in c1/c2 would show first.
    """
    sched = DiffusionSchedule.build(50, kind="cosine")
    model = _tiny_model()
    x = torch.randn(4, 3, 8, 128, generator=torch_generator("x0-eq-state")) * 2.0

    for t_index in (0, 1, 10, 25, 49):
        label = f"x0-eq-noise-{t_index}"
        got = p_sample_step(model, x, t_index, sched, x0_clamp=None,
                            generator=torch_generator(label))
        noise = torch.randn(x.shape, generator=torch_generator(label))
        want = _eps_form_step(model, x, t_index, sched, noise)
        assert torch.allclose(got, want, atol=1e-5), f"forms diverge at t={t_index}"


def test_x0_clamp_actually_binds():
    """With a clamp tight enough to bite, the step must differ from the raw form —
    otherwise the parameter is decorative and the blow-up fix is imaginary."""
    sched = DiffusionSchedule.build(50, kind="cosine")
    model = _tiny_model()
    # large state at the noisiest step => |x0_hat| = |x| / sqrt(alphabar) >> 1
    x = torch.randn(2, 3, 8, 128, generator=torch_generator("x0-clamp-state")) * 5.0
    label = "x0-clamp-noise"
    clamped = p_sample_step(model, x, 49, sched, x0_clamp=6.5,
                            generator=torch_generator(label))
    raw = p_sample_step(model, x, 49, sched, x0_clamp=None,
                        generator=torch_generator(label))
    assert not torch.allclose(clamped, raw)
    assert clamped.abs().max() < raw.abs().max()


def test_sample_generator_is_reproducible_without_global_seeding():
    """Two same-label generators => bit-identical chains, with the global RNG
    deliberately scrambled differently before each call. This is the property
    `scripts/run_scorecard.py` used to fake by pinning process-global state."""
    sched = DiffusionSchedule.build(20, kind="cosine")
    model = _tiny_model()
    torch.manual_seed(1)
    a = sample(model, (2, 3, 8, 128), sched, torch.device("cpu"),
               generator=torch_generator("sampler-repro"))
    torch.manual_seed(2)   # different global state; must not matter
    b = sample(model, (2, 3, 8, 128), sched, torch.device("cpu"),
               generator=torch_generator("sampler-repro"))
    assert torch.equal(a, b)

    c = sample(model, (2, 3, 8, 128), sched, torch.device("cpu"),
               generator=torch_generator("sampler-other-label"))
    assert not torch.equal(a, c), "different labels must fork different streams"


def test_smoke_checkpoint_samples_stay_bounded():
    """THE regression test for the sampler blow-up.

    The bug: sampling the SMOKE checkpoint produced image-space std ~1.9e3
    against a real 0.333 — a 6000x explosion that saturated every decoded pixel.
    The fix (x0_clamp at the data-derived 6.5, beta clip 0.999) does not make a
    4-epoch model GOOD; it makes it fail BOUNDED, and the bars here are the
    bounds the MECHANISM guarantees, not sample-quality hopes:

      * the final reverse step returns clamped x0_hat exactly (coef2 is 0 at
        t=0), so |pixel| <= x0_clamp is a hard ceiling — assert it exactly;
      * that ceiling caps std at x0_clamp = ~19.5x the real 0.333, so 30x real
        separates "bounded bad" from "exploded" with margin. Measured on the
        retrained 4-epoch checkpoint the clamped std is ~4.3 (a weak model
        railing against the clamp — the expected SMOKE result, recorded in
        RESEARCH.md Phase 4's correction, not tuned away); the broken sampler
        missed this bar by two orders of magnitude.
    """
    from diffmodel.config import CHECKPOINTS, get_config
    from diffmodel.pipeline import prepare

    ckpt = CHECKPOINTS / "smoke_latest.pt"
    if not ckpt.exists():
        pytest.skip("checkpoints/smoke_latest.pt not present (gitignored; train SMOKE first)")

    cfg = get_config("SMOKE")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = UNet(3, cfg.base_channels, cfg.channel_mults, cfg.num_res_blocks,
                 cfg.attention_at_bottleneck, dropout=cfg.dropout)
    model.load_state_dict(ck["ema"], strict=True)
    model.eval()

    sched = DiffusionSchedule.build(cfg.timesteps, kind=cfg.beta_schedule)
    out = sample(model, (64, *cfg.image_shape), sched, torch.device("cpu"),
                 x0_clamp=cfg.x0_clamp, generator=torch_generator("bounded-sample-test"))
    real_std = prepare(cfg, verbose=False).train_images.std()
    assert torch.isfinite(out).all()
    assert out.abs().max().item() <= cfg.x0_clamp + 1e-4, \
        "the final step must return the clamped x0_hat; the ceiling is exact"
    assert out.std().item() < 30 * real_std, (
        f"sampled std {out.std().item():.3g} vs real {real_std:.3g}: the sampler "
        f"is blowing up again"
    )


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
