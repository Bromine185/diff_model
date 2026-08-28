"""Fixture loading and day-image assembly.

Reads `data/fixtures/bars_5m.parquet` (produced once by `scripts/fetch_bars.py`)
and turns it into the (B, 3, 78) array the wavelet packing expects. Never touches
the network — see CLAUDE.md non-negotiable #3.

The three channels, in the paper's order:

    0  log return   R
    1  spread       G
    2  volume       B
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BARS_PER_DAY, FIXTURES

PARQUET = FIXTURES / "bars_5m.parquet"


# --------------------------------------------------------------------------
# Channel construction
# --------------------------------------------------------------------------

def log_returns(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """78 log returns from 78 bars.

    The obvious definition, r_t = log(c_t / c_{t-1}), gives only 77 values inside
    a session and needs the *previous session's* close for the first. We
    deliberately do not use that: the overnight gap is a different process,
    routinely 10-50x the size of a 5-minute intraday move, and including it would
    put a single enormous outlier in column 0 of every image. The model would
    learn "bar 0 is always huge", which is an artifact of our framing rather than
    intraday microstructure.

    Instead bar 0 uses its own open-to-close return, and the rest are close-to-close:

        r_0 = log(c_0 / o_0)
        r_t = log(c_t / c_{t-1})     t >= 1

    So the series is strictly within-session, which is what the paper's intraday
    seasonality claim is about.
    """
    r = np.empty_like(close, dtype=np.float64)
    r[..., 0] = np.log(close[..., 0] / open_[..., 0])
    r[..., 1:] = np.log(close[..., 1:] / close[..., :-1])
    return r


def spread_highlow(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Normalised high-low range, (h - l) / c.

    A PROXY. The paper uses true bid-ask spreads; yfinance 5-minute bars are
    OHLCV with no quote data. The range is not the spread — it is dominated by
    volatility within the bar rather than by the cost of crossing — but it shares
    the properties that matter for this reproduction: strictly positive, fat
    right tail, strong intraday U-shape, and positive co-movement with volume.

    This is the project's one substantive deviation from the paper. See
    `spread_corwin_schultz` for a closer estimator.
    """
    return (high - low) / close


def spread_corwin_schultz(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Corwin & Schultz (2012) high-low spread estimator, adapted to 5-min bars.

    Estimates the effective spread from the insight that the high-low range over
    two adjacent periods reflects both volatility (which scales with the time
    span) and the spread (which does not). Solving the two together separates them.

        beta  = E[ (log H_t/L_t)^2 + (log H_{t+1}/L_{t+1})^2 ]
        gamma = ( log( max(H_t,H_{t+1}) / min(L_t,L_{t+1}) ) )^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt2)  -  sqrt(gamma / (3 - 2*sqrt2))
        S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))

    Closer to a real spread than the raw range, at the cost of being noisier and
    frequently negative (the estimator is unbiased in expectation, not per
    observation). Negatives are floored at zero, which is what Corwin & Schultz
    themselves recommend. The last bar reuses the previous value, since the
    estimator needs a following bar.
    """
    two_sqrt2 = 3.0 - 2.0 * np.sqrt(2.0)
    h1, l1 = high[..., :-1], low[..., :-1]
    h2, l2 = high[..., 1:], low[..., 1:]

    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
    gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / two_sqrt2 - np.sqrt(gamma / two_sqrt2)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.maximum(s, 0.0)

    # pad the final bar (no t+1 available) by repeating the last estimate
    return np.concatenate([s, s[..., -1:]], axis=-1)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DayImages:
    """Assembled day-series plus the metadata needed for an honest split."""

    series: np.ndarray        # (B, 3, 78) float64 — returns, spread, volume
    tickers: np.ndarray       # (B,) str
    sessions: np.ndarray      # (B,) datetime64[D]

    def __len__(self) -> int:
        return len(self.series)

    def split_by_date(self, val_sessions: int) -> tuple["DayImages", "DayImages"]:
        """Hold out the LAST `val_sessions` calendar dates.

        Split by date, never randomly, and never by row. A random split would put
        the same trading day's AAPL in train and its MSFT in validation; those two
        share a market-wide volatility shock, so the validation set would be
        contaminated by construction and every metric would look better than it is.
        """
        uniq = np.unique(self.sessions)
        if val_sessions >= len(uniq):
            raise ValueError(f"val_sessions={val_sessions} but only {len(uniq)} sessions exist")
        cutoff = uniq[-val_sessions]
        is_val = self.sessions >= cutoff
        return self._subset(~is_val), self._subset(is_val)

    def _subset(self, mask: np.ndarray) -> "DayImages":
        return DayImages(self.series[mask], self.tickers[mask], self.sessions[mask])

    def take(self, n: int | None, rng: np.random.Generator | None = None) -> "DayImages":
        """First n (or a random n if `rng` given). Used by the SMOKE preset."""
        if n is None or n >= len(self):
            return self
        idx = rng.choice(len(self), size=n, replace=False) if rng else np.arange(n)
        return self._subset(np.sort(idx))


def load_bars(path: Path | None = None) -> pd.DataFrame:
    """Raw fixture. Fails with a pointed message rather than a FileNotFoundError."""
    p = path or PARQUET
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. The fixture is committed to the repo; if it is missing, "
            f"regenerate it once with:\n    python scripts/fetch_bars.py"
        )
    return pd.read_parquet(p)


def build_day_images(
    bars: pd.DataFrame | None = None,
    *,
    spread_method: str = "highlow",
) -> DayImages:
    """Pivot tidy bars into (B, 3, 78) day-series.

    Only (ticker, session) groups with exactly 78 bars are kept; `fetch_bars.py`
    already enforces that, and this re-checks rather than trusting it.
    """
    df = bars if bars is not None else load_bars()
    need = {"ticker", "ts", "session", "open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"fixture missing columns: {sorted(missing)}")

    df = df.sort_values(["ticker", "ts"])
    sizes = df.groupby(["ticker", "session"]).size()
    good = sizes[sizes == BARS_PER_DAY].index
    if len(good) == 0:
        raise ValueError("no (ticker, session) group has the expected 78 bars")

    df = df.set_index(["ticker", "session"]).loc[good].reset_index()
    df = df.sort_values(["ticker", "session", "ts"])

    n_groups = len(good)
    arr = {c: df[c].to_numpy().reshape(n_groups, BARS_PER_DAY) for c in ("open", "high", "low", "close", "volume")}

    if spread_method == "highlow":
        spread = spread_highlow(arr["high"], arr["low"], arr["close"])
    elif spread_method == "corwin_schultz":
        spread = spread_corwin_schultz(arr["high"], arr["low"])
    else:
        raise ValueError(f"unknown spread_method {spread_method!r}")

    series = np.stack(
        [
            log_returns(arr["open"], arr["close"]),
            spread,
            arr["volume"].astype(np.float64),
        ],
        axis=1,
    )

    keys = df.groupby(["ticker", "session"], sort=False).first().index
    tickers = np.array([k[0] for k in keys])
    sessions = np.array([np.datetime64(k[1], "D") for k in keys])

    # A single bad bar (a zero close, a split artefact) produces inf/nan here and
    # would silently poison the training set. Drop those day-images and say so.
    finite = np.isfinite(series).all(axis=(1, 2))
    if not finite.all():
        dropped = int((~finite).sum())
        print(f"build_day_images: dropped {dropped} day-image(s) with non-finite values")
        series, tickers, sessions = series[finite], tickers[finite], sessions[finite]

    order = np.lexsort((tickers, sessions))
    return DayImages(series[order], tickers[order], sessions[order])
