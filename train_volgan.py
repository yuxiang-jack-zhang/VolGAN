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
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

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
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Evaluate and (maybe) save best checkpoint every N epochs")
    parser.add_argument("--best-output", type=Path, default=None,
                        help="Path for best-checkpoint (default: <output>.best.pt)")
    parser.add_argument("--wandb-project", default="volgan",
                        help="W&B project name (pass --no-wandb to disable)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging even if wandb is installed")
    args = parser.parse_args()
    if args.best_output is None:
        args.best_output = args.output.with_suffix(".best.pt")

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

    # W&B
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            config={
                "epochs": args.epochs,
                "grad_epochs": args.grad_epochs,
                "noise_dim": args.noise_dim,
                "hidden_dim": args.hidden_dim,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "train_end": args.train_end,
                "seed": args.seed,
            },
        )
    elif not _WANDB_AVAILABLE:
        print("wandb not installed — logging to CSV only. Run: pip install wandb")

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

    # Main training loop — inlined from VolGAN.TrainLoopNoVal so we can log and checkpoint.
    print(f"\nTraining ({args.epochs} epochs) ...")

    # Pre-compute smoothness penalty matrices (copied from TrainLoopNoVal)
    lk, lt = 10, 8
    dtm = tau * 365
    mP_t, mP_k, mPb_K = VolGAN.penalty_mutau_tensor(m, dtm, device)
    moneyness_t = torch.tensor(m, dtype=torch.float, device=device)
    tau_t = torch.tensor(tau, dtype=torch.float, device=device)
    Ngrid = lk * lt
    t_seq = torch.zeros(tau_t.shape[0], dtype=torch.float, device=device)
    for i in range(tau_t.shape[0] - 1):
        t_seq[i] = 1 / ((tau_t[i + 1] - tau_t[i]) ** 2)
    matrix_t = torch.zeros((Ngrid, Ngrid), device=device, dtype=torch.float)
    for i in range(Ngrid - 1):
        matrix_t[i, i] = -1
        matrix_t[i, i + 1] = 1
    tsq = t_seq.repeat(lk).unsqueeze(0)
    matrix_m = torch.zeros((Ngrid - lk, Ngrid), device=device, dtype=torch.float)
    for i in range(Ngrid - lk):
        matrix_m[i, i] = -1
        matrix_m[i, i + lk] = 1
    m_seq = torch.zeros(lk * (lt - 1), dtype=torch.float, device=device)
    for i in range(moneyness_t.shape[0] - 1):
        m_seq[i * lk:(i + 1) * lk] = 1 / ((moneyness_t[i + 1] - moneyness_t[i]) ** 2)

    n_train = cond_t.shape[0]
    n_batches = n_train // args.batch_size + 1

    # Fixed held-out conditioning for eval (first 1000 test rows, or all if fewer)
    n_eval = min(1000, cond_test_t.shape[0])
    eval_cond = cond_test_t[:n_eval]
    true_ret_std = float(true_test_t[:, 0].std())

    # Log file for loss curves
    log_path = args.output.with_suffix(".log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "disc_loss", "gen_loss", "D_real", "D_fake",
                         "gen_ret_std", "ret_std_gap"])

    best_ret_std_gap = float("inf")
    best_epoch = -1

    def _save_checkpoint(path, epoch):
        torch.save({
            "gen_state": gen.state_dict(),
            "disc_state": disc.state_dict(),
            "noise_dim": args.noise_dim,
            "hidden_dim": args.hidden_dim,
            "cond_dim": cond_dim,
            "out_dim": out_dim,
            "train_end": args.train_end,
            "alpha": alpha,
            "beta": beta,
            "epoch": epoch,
            "true_test": true[~train_mask],
            "condition_test": condition[~train_mask],
            "test_dates": dates_t[~train_mask].dt.strftime("%Y-%m-%d").tolist(),
        }, path)

    gen.train()
    for epoch in tqdm(range(args.epochs)):
        perm = torch.randperm(n_train)
        cond_t = cond_t[perm]
        true_t = true_t[perm]

        epoch_disc_loss = epoch_gen_loss = epoch_d_real = epoch_d_fake = 0.0

        for i in range(n_batches):
            bs = args.batch_size if i < n_batches - 1 else n_train - i * args.batch_size
            if bs <= 0:
                continue
            cond_b = cond_t[i * args.batch_size: i * args.batch_size + bs]
            surf_b = cond_b[:, 3:]
            real_b = true_t[i * args.batch_size: i * args.batch_size + bs]

            # Discriminator update
            disc_opt.zero_grad()
            noise = torch.randn(bs, args.noise_dim, device=device)
            fake = gen(noise, cond_b)
            disc_real = disc(torch.cat([cond_b, real_b], dim=-1))
            disc_fake = disc(torch.cat([cond_b, fake], dim=-1).detach())
            disc_loss = (criterion(disc_fake, torch.zeros_like(disc_fake)) +
                         criterion(disc_real, torch.ones_like(disc_real))) / 2
            disc_loss.backward()
            disc_opt.step()

            # Generator update
            gen_opt.zero_grad()
            noise = torch.randn(bs, args.noise_dim, device=device)
            fake = gen(noise, cond_b)
            fake_surf = fake[:, 1:] + surf_b
            penalties_m = sum(
                torch.matmul(m_seq, torch.matmul(matrix_m, fake_surf[j]) ** 2)
                for j in range(bs)
            ) / bs
            penalties_t = sum(
                torch.matmul(tsq, torch.matmul(matrix_t, fake_surf[j]) ** 2)
                for j in range(bs)
            ) / bs
            disc_fake_pred = disc(torch.cat([cond_b, fake], dim=-1))
            gen_loss = (criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
                        + alpha * penalties_m + beta * penalties_t)
            gen_loss.backward()
            gen_opt.step()

            epoch_disc_loss += disc_loss.item()
            epoch_gen_loss += gen_loss.item()
            epoch_d_real += disc_real.mean().item()
            epoch_d_fake += disc_fake.mean().item()

        epoch_disc_loss /= n_batches
        epoch_gen_loss /= n_batches
        epoch_d_real /= n_batches
        epoch_d_fake /= n_batches

        # Evaluation every --eval-every epochs
        gen_ret_std = float("nan")
        ret_std_gap = float("nan")
        if (epoch + 1) % args.eval_every == 0:
            gen.eval()
            with torch.no_grad():
                eval_noise = torch.randn(n_eval, args.noise_dim, device=device)
                eval_out = gen(eval_noise, eval_cond)
            gen_ret_std = float(eval_out[:, 0].std().item())
            ret_std_gap = abs(gen_ret_std - true_ret_std)
            if ret_std_gap < best_ret_std_gap:
                best_ret_std_gap = ret_std_gap
                best_epoch = epoch + 1
                _save_checkpoint(args.best_output, epoch + 1)
                tqdm.write(f"  [epoch {epoch+1}] new best: gen_ret_std={gen_ret_std:.4f} "
                           f"(true={true_ret_std:.4f}, gap={ret_std_gap:.4f}) → {args.best_output}")

            if use_wandb:
                # Vol surface image: 3 samples from first eval conditioning
                first_cond = eval_cond[0:1].expand(3, -1)
                with torch.no_grad():
                    surf_samples = gen(torch.randn(3, args.noise_dim, device=device), first_cond)
                cur_log_iv = eval_cond[0, 3:].cpu().numpy()
                fig, axes = plt.subplots(1, 3, figsize=(12, 3),
                                         subplot_kw={"projection": "3d"})
                M, T = np.meshgrid(m, tau * 365)
                for ax, j in zip(axes, range(3)):
                    iv = np.exp(cur_log_iv + surf_samples[j, 1:].cpu().numpy())
                    iv = iv.reshape(lt, lk)  # [tau, moneyness]
                    ax.plot_surface(M, T, iv, cmap="viridis", alpha=0.85)
                    ax.set_xlabel("m"); ax.set_ylabel("τ (d)"); ax.set_zlabel("IV")
                fig.suptitle(f"Epoch {epoch+1} — sample surfaces")
                fig.tight_layout()
                wandb.log({
                    "gen_ret_std": gen_ret_std,
                    "ret_std_gap": ret_std_gap,
                    "sample_surfaces": wandb.Image(fig),
                }, step=epoch + 1)
                plt.close(fig)

            gen.train()

        metrics = {
            "disc_loss": epoch_disc_loss,
            "gen_loss": epoch_gen_loss,
            "D_real": epoch_d_real,
            "D_fake": epoch_d_fake,
        }
        if use_wandb:
            wandb.log(metrics, step=epoch + 1)
        log_writer.writerow([epoch + 1, f"{epoch_disc_loss:.6f}", f"{epoch_gen_loss:.6f}",
                             f"{epoch_d_real:.4f}", f"{epoch_d_fake:.4f}",
                             f"{gen_ret_std:.4f}" if not np.isnan(gen_ret_std) else "",
                             f"{ret_std_gap:.4f}" if not np.isnan(ret_std_gap) else ""])

    log_file.close()
    if use_wandb:
        wandb.finish()
    print(f"\nLoss log saved to {log_path}")
    print(f"Best checkpoint (epoch {best_epoch}, ret_std_gap={best_ret_std_gap:.4f}) → {args.best_output}")

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
