"""
Train VolGAN on the preprocessed OptionMetrics surfaces and save a checkpoint.

Usage:
  python train_volgan.py \\
      --data-dir data/volgan_prepared \\
      --train-end 2018-06-16 \\
      --output volgan_checkpoint.pt \\
      --device cuda \\
      --epochs 10000

The script bypasses VolGAN.py's SPXData() (which downloads from Yahoo Finance) and
loads the local CSV files produced by prepare_volgan_data.py instead.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

# pandas_datareader has a version-incompatibility bug; stub it so VolGAN.py can be imported.
# We only use Generator, Discriminator, GradientMatching, TrainLoopNoVal — none of which
# call SPXData() or pdr.get_data_yahoo().
import types, sys as _sys
if "pandas_datareader" not in _sys.modules:
    _pdr_stub = types.ModuleType("pandas_datareader")
    _pdr_stub.data = types.ModuleType("pandas_datareader.data")
    _sys.modules["pandas_datareader"] = _pdr_stub
    _sys.modules["pandas_datareader.data"] = _pdr_stub.data

# scipy.arange / array / exp were removed in newer scipy; VolGAN.py imports them at top level.
import scipy as _scipy, numpy as _np
if not hasattr(_scipy, "arange"):
    _scipy.arange = _np.arange
    _scipy.array = _np.array
    _scipy.exp = _np.exp

import VolGAN


def load_local_data(data_dir: Path, train_end: str):
    """
    Load the outputs of prepare_volgan_data.py and return the same quantities
    that VolGAN.DataPreprocesssing() produces, but using local SPX prices.

    Returns
    -------
    true : np.ndarray [N-22, 81]   (annualized log-return + 80 log-IV increments)
    condition : np.ndarray [N-22, 83]  (R_{t-1}, R_{t-2}, gamma_{t-1}, 80 log-IV at t-1)
    m : np.ndarray [10]   moneyness grid
    tau : np.ndarray [8]  tau grid in years
    ms, taus : np.ndarray [10,8] meshgrids
    dates_t : pd.DatetimeIndex  dates aligned with rows of true/condition (starts at index 22)
    train_mask : np.ndarray bool [N-22]  True for training rows
    """
    surfaces_transform = pd.read_csv(data_dir / "surfaces_transform.csv", index_col=0).values
    prices_df = pd.read_csv(data_dir / "spx_prices.csv", parse_dates=["date"])
    dates_df = pd.read_csv(data_dir / "dates.csv", parse_dates=["date"])
    m = np.load(data_dir / "moneyness_grid.npy")
    tau = np.load(data_dir / "tau_grid.npy")

    prices = prices_df["close"].values
    log_rtn = prices_df["log_return"].values
    dates_dt = pd.to_datetime(dates_df["date"])

    train_cutoff = pd.to_datetime(train_end)

    # VolGAN DataPreprocesssing() uses a 22-step warmup (21 lags for realized vol + 1)
    realised_vol_tm1 = np.array([
        np.sqrt(252 / 21) * np.sqrt(np.sum(log_rtn[i : i + 21] ** 2))
        for i in range(len(log_rtn) - 22)
    ])

    dates_t = dates_dt[22:]
    log_rtn_t = log_rtn[22:]
    log_rtn_tm1 = np.sqrt(252) * log_rtn[21:-1]
    log_rtn_tm2 = np.sqrt(252) * log_rtn[20:-2]

    log_iv_t = np.log(surfaces_transform[22:])
    log_iv_tm1 = np.log(surfaces_transform[21:-1])
    log_iv_inc_t = log_iv_t - log_iv_tm1

    log_rtn_t_ann = np.sqrt(252) * log_rtn_t

    # condition: [R_{t-1}, R_{t-2}, gamma_{t-1}, log_iv_{t-1}]  shape [N, 83]
    condition = np.concatenate(
        [
            np.expand_dims(log_rtn_tm1, 1),
            np.expand_dims(log_rtn_tm2, 1),
            np.expand_dims(realised_vol_tm1, 1),
            log_iv_tm1,
        ],
        axis=1,
    )
    # true: [annualized log-return, log-IV increments]  shape [N, 81]
    true = np.concatenate([np.expand_dims(log_rtn_t_ann, 1), log_iv_inc_t], axis=1)

    train_mask = dates_t <= train_cutoff
    taus, ms = np.meshgrid(tau, m)  # ms[i,j] = m[i], taus[i,j] = tau[j]

    return true, condition, m, tau, ms, taus, dates_t, train_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/volgan_prepared"))
    parser.add_argument("--train-end", default="2018-06-16",
                        help="Last date (inclusive) to use for training")
    parser.add_argument("--output", type=Path, default=Path("volgan_checkpoint.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--grad-epochs", type=int, default=25,
                        help="Gradient matching warmup epochs")
    parser.add_argument("--noise-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    print(f"Loading data from {args.data_dir} ...")
    true, condition, m, tau, ms, taus, dates_t, train_mask = load_local_data(
        args.data_dir, args.train_end
    )

    n_train = train_mask.sum()
    n_test = (~train_mask).sum()
    print(f"Training rows : {n_train}  ({dates_t[train_mask].min().date()} – {dates_t[train_mask].max().date()})")
    print(f"Test rows     : {n_test}  ({dates_t[~train_mask].min().date()} – {dates_t[~train_mask].max().date()})")

    true_t = torch.from_numpy(true[train_mask]).float().to(device)
    cond_t = torch.from_numpy(condition[train_mask]).float().to(device)
    true_test_t = torch.from_numpy(true[~train_mask]).float().to(device)
    cond_test_t = torch.from_numpy(condition[~train_mask]).float().to(device)

    cond_dim = cond_t.shape[1]   # 83
    out_dim = true_t.shape[1]    # 81

    gen = VolGAN.Generator(
        noise_dim=args.noise_dim,
        cond_dim=cond_dim,
        hidden_dim=args.hidden_dim,
        output_dim=out_dim,
    ).to(device)
    disc = VolGAN.Discriminator(
        in_dim=cond_dim + out_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    gen_opt = torch.optim.RMSprop(gen.parameters(), lr=args.lr)
    disc_opt = torch.optim.RMSprop(disc.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss().to(device)

    # Gradient matching to set smoothness penalty weights alpha, beta
    print(f"\nGradient matching ({args.grad_epochs} epochs) ...")
    gen, gen_opt, disc, disc_opt, criterion, alpha, beta = VolGAN.GradientMatching(
        gen, gen_opt, disc, disc_opt, criterion,
        cond_t, true_t,
        m, tau, ms, taus,
        args.grad_epochs, args.lr, args.lr,
        args.batch_size, args.noise_dim, device,
        lk=10, lt=8,
    )

    # Main training loop
    print(f"\nTraining ({args.epochs} epochs) ...")
    gen, gen_opt, disc, disc_opt, criterion = VolGAN.TrainLoopNoVal(
        alpha, beta,
        gen, gen_opt, disc, disc_opt, criterion,
        cond_t, true_t,
        m, tau, ms, taus,
        args.epochs, args.lr, args.lr,
        args.batch_size, args.noise_dim, device,
        lk=10, lt=8,
    )

    checkpoint = {
        "gen_state": gen.state_dict(),
        "disc_state": disc.state_dict(),
        "noise_dim": args.noise_dim,
        "hidden_dim": args.hidden_dim,
        "cond_dim": cond_dim,
        "out_dim": out_dim,
        "train_end": args.train_end,
        "alpha": alpha,
        "beta": beta,
        # save test split for downstream use
        "true_test": true[~train_mask],
        "condition_test": condition[~train_mask],
        "test_dates": dates_t[~train_mask].dt.strftime("%Y-%m-%d").tolist(),
    }
    torch.save(checkpoint, args.output)
    print(f"\nCheckpoint saved to {args.output}")


if __name__ == "__main__":
    main()
