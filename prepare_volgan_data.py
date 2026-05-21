"""
Prepare VolGAN training data from raw OptionMetrics files.

The raw options already have impl_volatility, moneyness, ttm, and vega computed,
so this script applies vega-weighted Nadaraya-Watson smoothing directly.

Outputs (in --output-dir, default data/volgan_prepared/):
  surfaces_transform.csv  -- one row per trading day, 80 cols (raw IV, tau-major flat)
  spx_prices.csv          -- date, close, log_return (from local underlying files)
  dates.csv               -- dates aligned row-for-row with surfaces (YYYY-MM-DD)
  moneyness_grid.npy      -- the 10 moneyness points used
  tau_grid.npy            -- the 8 tau values in years used

Grid (must match what VolGAN.py uses at inference time):
  moneyness: np.linspace(0.6, 1.4, 10)
  tau_days:  [7, 14, 30, 60, 91, 182, 273, 365]

Flat layout: position i*10+j = (tau_i, moneyness_j), matching the VolGAN example:
  for i in range(8): fk[:,:,i] = surface[:, i*10:(i+1)*10]

Run time: ~20-40 min for 2000-2023 on a laptop.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from datacleaning import SmoothVega

DATA_ROOT = Path(
    "/Users/jackzhang/research/volatility-surface-simulation"
    "/data/VolGAN_optionmetrics_spx_20000103_20230228"
)

# VolGAN grid — keep in sync with VolGAN.py which uses np.linspace(0.6, 1.4, 10)
MONEYNESS_GRID = np.linspace(0.6, 1.4, 10)
TAU_DAYS = np.array([7, 14, 30, 60, 91, 182, 273, 365])
TAU_GRID = TAU_DAYS / 365.0

# Bandwidth from VolGAN paper Eq. 21 optimal values
# kernelVega uses exp(-x^2 / (2*h)), so h is the variance (not std)
NW_H1 = 0.002   # moneyness variance
NW_H2 = 0.046   # tau variance
MIN_QUOTES = 20  # skip days with fewer OTM quotes than this


def load_underlying(data_root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((data_root / "underlying").glob("spx_secprd_*.csv.gz")):
        df = pd.read_csv(f, parse_dates=["date"])
        frames.append(df[["date", "close", "return"]])
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    # OptionMetrics 'return' is daily log-return
    df = df.rename(columns={"return": "log_return"})
    return df


def smooth_day(grp: pd.DataFrame, h1: float, h2: float) -> np.ndarray | None:
    """
    Vega-weighted NW smoothing for one day's OTM options.
    Returns raw IV surface shape [10 (moneyness), 8 (tau)], or None if insufficient data.
    """
    otm = grp[
        grp["impl_volatility"].notna()
        & (grp["impl_volatility"] > 0)
        & grp["vega"].notna()
        & (grp["vega"] > 0)
        & grp["moneyness"].between(0.5, 1.5)
        & (grp["ttm"] > 0)
        & (grp["ttm"] <= 1.5)
    ]
    # OTM only: puts below ATM, calls at/above ATM
    otm = otm[
        ((otm["cp_flag"] == "P") & (otm["moneyness"] < 1.0))
        | ((otm["cp_flag"] == "C") & (otm["moneyness"] >= 1.0))
    ]
    if len(otm) < MIN_QUOTES:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surface = SmoothVega(
            otm["impl_volatility"].values,
            otm["moneyness"].values,
            otm["ttm"].values,
            MONEYNESS_GRID,
            TAU_GRID,
            h1,
            h2,
            otm["vega"].values,
        )

    if not np.all(np.isfinite(surface)) or np.any(surface <= 0):
        return None
    return surface  # [10, 8]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/volgan_prepared"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--h1", type=float, default=NW_H1)
    parser.add_argument("--h2", type=float, default=NW_H2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading underlying prices...")
    underlying = load_underlying(args.data_root)
    spot_map = underlying.set_index("date")["close"].to_dict()
    ret_map = underlying.set_index("date")["log_return"].to_dict()

    surfaces, dates_out = [], []
    years = range(args.start_year, args.end_year + 1)

    for year in years:
        fpath = args.data_root / "raw_options" / f"spx_options_{year}.csv.gz"
        if not fpath.exists():
            continue
        opts = pd.read_csv(fpath, parse_dates=["date"])
        for date, grp in tqdm(opts.groupby("date"), desc=str(year), leave=False):
            surface = smooth_day(grp, args.h1, args.h2)
            if surface is None:
                continue
            surfaces.append(surface)
            dates_out.append(date)

    n = len(surfaces)
    print(f"\nSuccessfully processed {n} trading days")

    # Stack to [N, 10 (moneyness), 8 (tau)]
    arr = np.stack(surfaces)

    # Flatten to [N, 80] in tau-major order so that position i*10+j = (tau_i, moneyness_j).
    # This matches the VolGAN example: fk[:,:,i] = surface[:, i*10:(i+1)*10]
    flat = arr.transpose(0, 2, 1).reshape(n, 80)  # [N, 8 (tau), 10 (m)] → [N, 80]

    # surfaces_transform.csv: index col + 80 IV cols (raw IV, not log-IV)
    sdf = pd.DataFrame(flat, columns=[f"iv_{i}" for i in range(80)])
    sdf.to_csv(args.output_dir / "surfaces_transform.csv")
    print(f"Saved surfaces_transform.csv  shape {sdf.shape}")

    # dates.csv
    pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates_out]}).to_csv(
        args.output_dir / "dates.csv", index=False
    )

    # spx_prices.csv: aligned with surface rows
    closes = [spot_map.get(d, np.nan) for d in dates_out]
    rets = [ret_map.get(d, np.nan) for d in dates_out]
    pd.DataFrame(
        {"date": [d.strftime("%Y-%m-%d") for d in dates_out], "close": closes, "log_return": rets}
    ).to_csv(args.output_dir / "spx_prices.csv", index=False)
    print(f"Saved spx_prices.csv")

    np.save(args.output_dir / "moneyness_grid.npy", MONEYNESS_GRID)
    np.save(args.output_dir / "tau_grid.npy", TAU_GRID)
    print(f"Saved grid files  m={MONEYNESS_GRID.round(3)}, tau_days={TAU_DAYS}")
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
