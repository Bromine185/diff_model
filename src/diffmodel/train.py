"""Training loop with checkpoint/resume.

Resume is not a nicety here. FULL is a 100-epoch run on a Colab GPU, and Colab
disconnects — a run that loses everything at hour three is the default outcome
without this. Checkpoints carry model, EMA, optimizer and epoch, so a resumed run
continues rather than restarting with a warm model and a cold optimizer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .ddpm import DiffusionSchedule, EMA, p_losses
from .seeding import torch_generator


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "epoch_seconds": self.epoch_seconds,
        }


def make_loader(images: np.ndarray, cfg: Config, label: str, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(images)).float())
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(ds) > cfg.batch_size,
        generator=torch_generator(label) if shuffle else None,
        num_workers=0,   # the dataset is already in memory; workers only add overhead
    )


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, loader: DataLoader, sched: DiffusionSchedule, device: torch.device,
    label: str = "val-eval",
) -> float:
    """Mean epsilon-prediction MSE over the loader.

    Uses a FIXED per-epoch seed for the timestep draw. Without that, validation
    loss jitters purely because different random t were sampled, and epoch-to-epoch
    comparison becomes noise.
    """
    model.eval()
    g = torch.Generator(device="cpu")
    g.manual_seed(1234)
    total, n = 0.0, 0
    for (x0,) in loader:
        x0 = x0.to(device)
        t = torch.randint(0, len(sched), (x0.shape[0],), generator=g).to(device)
        loss = p_losses(model, x0, t, sched)
        total += loss.item() * x0.shape[0]
        n += x0.shape[0]
    return total / max(n, 1)


def save_checkpoint(path: Path, model, ema, opt, epoch: int, cfg: Config, history: History) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": opt.state_dict(),
            "history": history.as_dict(),
            "config_name": cfg.name,
        },
        path,
    )


def load_checkpoint(path: Path, model, ema, opt, device: torch.device) -> tuple[int, History]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
    if opt is not None and "optimizer" in ck:
        opt.load_state_dict(ck["optimizer"])
    h = History(**ck.get("history", {}))
    return ck["epoch"], h


def train(
    model: nn.Module,
    train_images: np.ndarray,
    cfg: Config,
    device: torch.device,
    val_images: np.ndarray | None = None,
    sched: DiffusionSchedule | None = None,
    resume: bool = True,
    progress: bool = True,
) -> tuple[nn.Module, EMA, History, DiffusionSchedule]:
    """Train the noise predictor. Returns (model, ema, history, schedule)."""
    sched = sched or DiffusionSchedule.build(cfg.timesteps, kind=cfg.beta_schedule)
    sched = sched.to(device)
    model = model.to(device)
    ema = EMA(model, decay=cfg.ema_decay)
    ema.shadow.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ckpt = Path(cfg.ckpt_dir) / f"{cfg.name.lower()}_latest.pt"
    start_epoch, history = 0, History()
    if resume and ckpt.exists():
        start_epoch, history = load_checkpoint(ckpt, model, ema, opt, device)
        print(f"resumed from {ckpt} at epoch {start_epoch}")

    train_loader = make_loader(train_images, cfg, "train-shuffle", shuffle=True)
    val_loader = make_loader(val_images, cfg, "val", shuffle=False) if val_images is not None else None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"training {cfg.name}: {len(train_images):,} images, {n_params/1e6:.2f}M params, "
          f"T={cfg.timesteps} ({cfg.beta_schedule}), {cfg.epochs} epochs on {device}")

    # The EMA time constant is ~1/(1-decay) steps. If the run is short relative to
    # that, the shadow never leaves its initialisation and every sample is drawn
    # from an untrained network — with no error, just silently meaningless output.
    total_steps = max(len(train_loader), 1) * (cfg.epochs - start_epoch)
    retained = cfg.ema_decay ** total_steps
    if retained > 0.2:
        import warnings
        tau = 1.0 / max(1.0 - cfg.ema_decay, 1e-12)
        warnings.warn(
            f"EMA decay {cfg.ema_decay} has a time constant of ~{tau:.0f} steps but this run "
            f"is only {total_steps} steps: the shadow will retain {retained:.0%} of its "
            f"initial weights, so samples drawn from it will be near-untrained. "
            f"Lower ema_decay for short runs.",
            stacklevel=2,
        )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0

        iterator = train_loader
        if progress:
            from tqdm.auto import tqdm
            iterator = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", leave=False)

        for (x0,) in iterator:
            x0 = x0.to(device)
            t = torch.randint(0, len(sched), (x0.shape[0],), device=device)
            loss = p_losses(model, x0, t, sched)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ema.update(model)

            running += loss.item() * x0.shape[0]
            seen += x0.shape[0]

        history.train_loss.append(running / max(seen, 1))
        history.epoch_seconds.append(time.time() - t0)
        if val_loader is not None:
            history.val_loss.append(evaluate_loss(ema.shadow, val_loader, sched, device))

        msg = (f"epoch {epoch+1:3d}/{cfg.epochs}  train {history.train_loss[-1]:.4f}"
               f"  {history.epoch_seconds[-1]:.1f}s")
        if history.val_loss:
            msg += f"  val(ema) {history.val_loss[-1]:.4f}"
        print(msg, flush=True)

        if (epoch + 1) % cfg.ckpt_every == 0 or epoch + 1 == cfg.epochs:
            save_checkpoint(ckpt, model, ema, opt, epoch + 1, cfg, history)

    return model, ema, history, sched
