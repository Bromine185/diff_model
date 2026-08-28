#!/usr/bin/env python
"""Fetch 5-minute bars once, commit them, never touch the network again.

This is the ONLY file in the repo permitted to make a network call. Everything
downstream — notebooks, training, evaluation — reads `data/fixtures/bars_5m.parquet`.
That is what makes the Colab run train on byte-identical data to the local smoke
run, and what makes the notebooks executable offline.

Usage
-----
    python scripts/fetch_bars.py                 # full universe (~320 tickers)
    python scripts/fetch_bars.py --limit 12      # smoke: a handful of names
    python scripts/fetch_bars.py --chunk-size 25 # smaller batches if rate-limited

Constraint worth knowing: yfinance caps `interval="5m"` at `period="60d"`. There
is no parameter that extends this. We therefore buy sample count with BREADTH
(many tickers) rather than history, which means the fixture covers a single
macro regime. That limitation is real and is documented in CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from diffmodel.universe import universe  # noqa: E402

FIXTURES = REPO / "data" / "fixtures"
OUT_PARQUET = FIXTURES / "bars_5m.parquet"
OUT_MANIFEST = FIXTURES / "manifest.json"

# US regular session: 09:30 through 15:55 inclusive, 5-minute bars.
BARS_PER_DAY = 78
SESSION_TZ = "America/New_York"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def fetch_chunk(tickers: list[str], attempts: int = 3) -> pd.DataFrame | None:
    """Download one batch, retrying with backoff. Returns None if all attempts fail."""
    import yfinance as yf

    for attempt in range(1, attempts + 1):
        try:
            df = yf.download(
                tickers,
                period="60d",
                interval="5m",
                group_by="ticker",
                auto_adjust=False,
                prepost=False,      # regular session only; pre/post bars are thin and break the 78-bar grid
                progress=False,
                threads=True,
            )
            if df is not None and not df.empty:
                return df
            print(f"    empty frame (attempt {attempt}/{attempts})")
        except Exception as exc:  # noqa: BLE001 - report and retry whatever yfinance raises
            print(f"    {type(exc).__name__}: {exc} (attempt {attempt}/{attempts})")
        if attempt < attempts:
            time.sleep(2 ** attempt)
    return None


def tidy(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """MultiIndex-column yfinance frame -> long format, session-filtered.

    Keeps only days with exactly BARS_PER_DAY bars. Half-days (Thanksgiving,
    Christmas Eve, July 3rd) produce short sessions, and a 78-bar wavelet image
    cannot be built from 42 bars. Dropping them is cleaner than padding, which
    would invent price action that never happened.
    """
    frames = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t][OHLCV].dropna()
        if sub.empty:
            continue
        sub = sub.copy()
        sub.index = sub.index.tz_convert(SESSION_TZ)
        sub["session"] = sub.index.date

        counts = sub.groupby("session").size()
        complete = counts[counts == BARS_PER_DAY].index
        sub = sub[sub["session"].isin(set(complete))]
        if sub.empty:
            continue

        sub.insert(0, "ticker", t)
        frames.append(sub.reset_index(names="ts"))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).lower() for c in out.columns]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="truncate the universe (smoke runs)")
    ap.add_argument("--chunk-size", type=int, default=40, help="tickers per yfinance request")
    args = ap.parse_args()

    tickers = universe(limit=args.limit)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    print(f"fetching {len(tickers)} tickers, 5m bars, 60d, in chunks of {args.chunk_size}")

    collected: list[pd.DataFrame] = []
    failed_chunks: list[list[str]] = []

    for i in range(0, len(tickers), args.chunk_size):
        chunk = tickers[i : i + args.chunk_size]
        n = i // args.chunk_size + 1
        total = (len(tickers) + args.chunk_size - 1) // args.chunk_size
        print(f"  chunk {n}/{total}: {chunk[0]}..{chunk[-1]} ({len(chunk)})", flush=True)

        raw = fetch_chunk(chunk)
        if raw is None:
            print("    FAILED after retries")
            failed_chunks.append(chunk)
            continue
        part = tidy(raw, chunk)
        if part.empty:
            print("    no complete sessions survived filtering")
            continue
        collected.append(part)
        print(f"    kept {part['ticker'].nunique()} tickers, {len(part):,} bars", flush=True)

    if not collected:
        print("\nFATAL: nothing fetched.", file=sys.stderr)
        return 1

    bars = pd.concat(collected, ignore_index=True).sort_values(["ticker", "ts"]).reset_index(drop=True)

    # Cast down: float32 is far more precision than 5-minute prices carry, and it
    # halves the committed fixture. Volume stays integral but can exceed int32.
    for c in ["open", "high", "low", "close"]:
        bars[c] = bars[c].astype("float32")
    bars["volume"] = bars["volume"].astype("int64")

    bars.to_parquet(OUT_PARQUET, index=False, compression="zstd")

    sessions = bars.groupby("ticker")["session"].nunique()
    manifest = {
        "fetched_at_utc": datetime.now(UTC).isoformat(),
        "yfinance_version": __import__("yfinance").__version__,
        "interval": "5m",
        "period": "60d",
        "bars_per_day": BARS_PER_DAY,
        "session_tz": SESSION_TZ,
        "tickers_requested": len(tickers),
        "tickers_kept": int(bars["ticker"].nunique()),
        "tickers_dropped": sorted(set(tickers) - set(bars["ticker"].unique())),
        "failed_chunks": failed_chunks,
        "sessions_min": int(sessions.min()),
        "sessions_max": int(sessions.max()),
        "sessions_median": int(sessions.median()),
        "date_min": str(bars["session"].min()),
        "date_max": str(bars["session"].max()),
        "rows": int(len(bars)),
        "day_images_available": int(len(bars) // BARS_PER_DAY),
        "parquet_bytes": OUT_PARQUET.stat().st_size,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\nwrote {OUT_PARQUET.relative_to(REPO)}  "
          f"({manifest['parquet_bytes'] / 1e6:.1f} MB)")
    print(f"  {manifest['tickers_kept']} tickers x ~{manifest['sessions_median']} sessions "
          f"= {manifest['day_images_available']:,} day-images")
    print(f"  {manifest['date_min']} .. {manifest['date_max']}")
    if manifest["tickers_dropped"]:
        print(f"  dropped {len(manifest['tickers_dropped'])}: "
              f"{', '.join(manifest['tickers_dropped'][:15])}"
              f"{' ...' if len(manifest['tickers_dropped']) > 15 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
