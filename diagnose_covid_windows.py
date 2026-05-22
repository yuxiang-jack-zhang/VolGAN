#!/usr/bin/env python3
"""Diagnose why Covid-period windows fail in build_instrument_panel.

For each monthly window candidate Feb–Jul 2020, this script:
  1. Attempts build_instrument_panel and reports the exact exception.
  2. Checks raw option row counts and paired-strike availability
     for the resolved start date and the next 2 trading dates.

Usage:
    conda activate diffusion
    python VolGAN/diagnose_covid_windows.py \
        --data-dir volatility-surface-simulation/data/VolGAN_optionmetrics_spx_20000103_20230228 \
        [--m0 0.9]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "volatility-surface-simulation"))
from hedging import (
    load_underlying,
    load_raw_options,
    first_trading_date_on_or_after,
    choose_expiry,
    select_target_straddle,
    build_instrument_panel,
    _as_timestamp,
)


def _check_date(date: pd.Timestamp, underlying: pd.DataFrame,
                data_dir: Path, m0: float, target_days: int = 30) -> dict:
    """Return a diagnostic dict for one candidate start date."""
    row = {"candidate": date.date(), "n_rows": 0, "n_expiries": 0,
           "n_paired_strikes": 0, "panel_ok": False, "error": None}
    opts = load_raw_options(data_dir=data_dir, start_date=date, end_date=date)
    day_rows = opts[opts["date"] == date]
    row["n_rows"] = len(day_rows)
    if day_rows.empty:
        row["error"] = "no option rows on this date"
        return row

    spot_series = underlying.loc[underlying["date"] == date, "close"]
    if spot_series.empty:
        row["error"] = "no spot on this date"
        return row
    spot = float(spot_series.iloc[0])

    try:
        expiry = choose_expiry(day_rows, target_days=target_days)
    except ValueError as e:
        row["error"] = f"choose_expiry: {e}"
        return row

    expiry_rows = day_rows[day_rows["exdate"] == expiry]
    row["n_expiries"] = day_rows["exdate"].nunique()

    paired = (
        expiry_rows.groupby("strike")["cp_flag"]
        .agg(lambda flags: {"C", "P"}.issubset(set(flags)))
        .loc[lambda has: has]
    )
    row["n_paired_strikes"] = len(paired)

    try:
        select_target_straddle(day_rows, spot, expiry, m0=m0)
        row["panel_ok"] = True
    except ValueError as e:
        row["error"] = f"select_target_straddle: {e}"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("volatility-surface-simulation/data/"
                                    "VolGAN_optionmetrics_spx_20000103_20230228"))
    parser.add_argument("--m0", type=float, default=0.9)
    args = parser.parse_args()

    covid_candidates = pd.date_range("2020-02-01", "2020-07-01", freq="MS")
    underlying = load_underlying(data_dir=args.data_dir,
                                 start_date="2020-01-15", end_date="2020-08-01")
    trading_dates = pd.DatetimeIndex(underlying["date"].drop_duplicates().sort_values())

    print(f"{'='*72}")
    print(f"Covid window diagnosis  m0={args.m0}  data-dir={args.data_dir}")
    print(f"{'='*72}\n")

    for candidate in covid_candidates:
        print(f"── {candidate.date()} ──────────────────────────────────────────")

        # First try build_instrument_panel directly
        try:
            panel = build_instrument_panel(candidate, m0=args.m0,
                                           data_dir=args.data_dir)
            print(f"  build_instrument_panel: OK  "
                  f"(start={panel.start_date.date()}, "
                  f"expiry={panel.expiry_date.date()}, "
                  f"n_hedges={len(panel.hedges)})")
        except Exception as e:
            print(f"  build_instrument_panel FAILED: [{type(e).__name__}] {e}")

        # Check the next 3 trading dates in detail
        try:
            actual_start = first_trading_date_on_or_after(candidate, underlying)
        except ValueError:
            print("  No trading date on or after candidate — skipping detail check.\n")
            continue

        start_idx = trading_dates.get_loc(actual_start)
        check_dates = [trading_dates[start_idx + k]
                       for k in range(3)
                       if start_idx + k < len(trading_dates)]

        rows = [_check_date(d, underlying, args.data_dir, args.m0)
                for d in check_dates]

        header = f"  {'date':12s} {'rows':>6} {'expiries':>9} {'paired':>7} {'ok':>4}  error"
        print(header)
        print(f"  {'-'*70}")
        for r in rows:
            ok_str = "YES" if r["panel_ok"] else "no"
            err = r["error"] or ""
            print(f"  {str(r['candidate']):12s} {r['n_rows']:>6} "
                  f"{r['n_expiries']:>9} {r['n_paired_strikes']:>7} "
                  f"{ok_str:>4}  {err}")
        print()


if __name__ == "__main__":
    main()
