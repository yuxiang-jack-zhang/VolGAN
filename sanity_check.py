"""
VolGAN checkpoint sanity check.

Verifies /Users/jackzhang/research/VolGAN/volgan_checkpoint.pt against the
Cont & Vuletić VolGAN paper (Tables 1–7).  Run with:

    conda activate diffusion
    python sanity_check.py

Checks
------
1  Metadata keys & architecture dimensions
2  Generator / discriminator state-dict layer shapes
3  Forward-pass: shape, no NaN/Inf, plausible output range
4  Return-distribution exceedance ratios   (Table 2, §4.2)
5  ATM 3-month IV distribution exceedance  (Table 2)
6  PCA alignment of IV surface increments  (Tables 4–5)
7  Leverage-effect correlation             (Tables 6–7)
8  Visual vol-surface plot → sanity_check_surfaces.png
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

# ── VolGAN import shims (mirrors volgan_adapter.py) ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

if "pandas_datareader" not in sys.modules:
    _stub = types.ModuleType("pandas_datareader")
    _stub.data = types.ModuleType("pandas_datareader.data")  # type: ignore[attr-defined]
    sys.modules["pandas_datareader"] = _stub
    sys.modules["pandas_datareader.data"] = _stub.data

import scipy as _scipy
if not hasattr(_scipy, "arange"):
    _scipy.arange = np.arange
    _scipy.array = np.array
    _scipy.exp = np.exp

import VolGAN  # noqa: E402 (after shims)

# ── Grid (mirrors volgan_adapter.py) ─────────────────────────────────────────
MONEYNESS_GRID = np.linspace(0.6, 1.4, 10)          # 10 moneyness points
TAU_DAYS = np.array([7, 14, 30, 60, 91, 182, 273, 365])
TAU_GRID = TAU_DAYS / 365.0                          # 8 tau points

CKPT_PATH = Path(__file__).parent / "volgan_checkpoint.pt"
PLOT_PATH = Path(__file__).parent / "sanity_check_surfaces.png"
SAMPLES_PER_DAY = 10   # K draws per test-day conditioning; ~10K total samples

# ── Helpers ───────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []

def report(name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    _results.append((name, passed, detail))
    print(f"  [{status}] {name}: {detail}")


def exceedance_ratio(generated: np.ndarray, true_quantile: float) -> float:
    """Fraction of generated samples below `true_quantile`."""
    return float(np.mean(generated < true_quantile))


def top_k_pcs(X: np.ndarray, k: int = 3):
    """Return top-k principal components of X (each column = one PC)."""
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / len(X)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order[:k]]


def variance_explained(eigenvalues: np.ndarray) -> np.ndarray:
    return eigenvalues / eigenvalues.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

# ─────────────────────────────────────────────────────────────────────────────
# Check 1 — Metadata / architecture dimensions
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 1: Metadata ──")
EXPECTED_META = {
    "noise_dim": 32,
    "hidden_dim": 16,
    "cond_dim": 83,
    "out_dim": 81,
}
for key, expected in EXPECTED_META.items():
    if key not in ckpt:
        report(f"meta/{key}", False, f"key missing from checkpoint")
    else:
        val = ckpt[key]
        ok = (val == expected)
        report(f"meta/{key}", ok, f"got {val}, expected {expected}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 2 — State-dict layer shapes
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 2: Layer shapes ──")

noise_dim  = ckpt.get("noise_dim",  32)
hidden_dim = ckpt.get("hidden_dim", 16)
cond_dim   = ckpt.get("cond_dim",   83)
out_dim    = ckpt.get("out_dim",    81)

gen = VolGAN.Generator(
    noise_dim=noise_dim, cond_dim=cond_dim,
    hidden_dim=hidden_dim, output_dim=out_dim,
)
gen.load_state_dict(ckpt["gen_state"])
gen.eval()

disc = VolGAN.Discriminator(
    in_dim=cond_dim + out_dim,
    hidden_dim=hidden_dim,
)
disc.load_state_dict(ckpt["disc_state"])
disc.eval()

GEN_SHAPES = {
    "linear1.weight": (hidden_dim, noise_dim + cond_dim),
    "linear2.weight": (hidden_dim * 2, hidden_dim),
    "linear3.weight": (out_dim, hidden_dim * 2),
}
DISC_SHAPES = {
    "linear1.weight": (hidden_dim, cond_dim + out_dim),
    "linear2.weight": (1, hidden_dim),
}

for layer, expected_shape in GEN_SHAPES.items():
    actual = tuple(ckpt["gen_state"][layer].shape)
    report(f"gen/{layer}", actual == expected_shape,
           f"shape {actual} (expected {expected_shape})")

for layer, expected_shape in DISC_SHAPES.items():
    actual = tuple(ckpt["disc_state"][layer].shape)
    report(f"disc/{layer}", actual == expected_shape,
           f"shape {actual} (expected {expected_shape})")

# ─────────────────────────────────────────────────────────────────────────────
# Check 3 — Forward-pass sanity
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 3: Forward pass ──")

cond_test = ckpt["condition_test"]   # [N_test, 83] — held-out test conditioning
if isinstance(cond_test, np.ndarray):
    cond_test = torch.from_numpy(cond_test).float()

first_cond = cond_test[0:1].expand(1000, -1)  # broadcast to [1000, 83]
noise = torch.randn(1000, noise_dim)

with torch.no_grad():
    out = gen(noise, first_cond)    # [1000, 81]

report("fwd/shape",     tuple(out.shape) == (1000, 81), str(tuple(out.shape)))
report("fwd/no_nan",    not torch.isnan(out).any().item(), "")
report("fwd/no_inf",    not torch.isinf(out).any().item(), "")

ret_std = out[:, 0].std().item()
report("fwd/ret_std",   0.005 < ret_std < 0.15,
       f"annualized-return std = {ret_std:.4f} (expected 0.005–0.15)")

iv_std = out[:, 1:].std().item()
report("fwd/iv_std",    iv_std < 2.0,
       f"log-IV-increment std = {iv_std:.4f} (expected < 2.0)")

# ─────────────────────────────────────────────────────────────────────────────
# Generate samples for checks 4–7
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Sampling…")

N_test = len(cond_test)
all_gen = []

with torch.no_grad():
    for i in range(N_test):
        cond_i = cond_test[i:i+1].expand(SAMPLES_PER_DAY, -1)
        noise_i = torch.randn(SAMPLES_PER_DAY, noise_dim)
        all_gen.append(gen(noise_i, cond_i).numpy())

gen_samples = np.concatenate(all_gen, axis=0)   # [N_test*K, 81]

true_test = ckpt["true_test"]
if isinstance(true_test, torch.Tensor):
    true_test = true_test.numpy()

# De-annualize returns (generator outputs annualized = sqrt(252) * daily_log_ret)
gen_ret   = gen_samples[:, 0] / np.sqrt(252)    # [N_test*K]
true_ret  = true_test[:, 0]   / np.sqrt(252)    # [N_test]

# ─────────────────────────────────────────────────────────────────────────────
# Check 4 — Return distribution (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 4: Return distribution exceedance ratios ──")

# True quantiles from the test set
q_ret_001  = np.quantile(true_ret, 0.010)
q_ret_0025 = np.quantile(true_ret, 0.025)
q_ret_0975 = np.quantile(true_ret, 0.975)
q_ret_099  = np.quantile(true_ret, 0.990)

er_001  = exceedance_ratio(gen_ret, q_ret_001)
er_0025 = exceedance_ratio(gen_ret, q_ret_0025)
er_0975 = exceedance_ratio(gen_ret, q_ret_0975)
er_099  = exceedance_ratio(gen_ret, q_ret_099)

# Paper Table 2 bounds (±~5% tolerance on reported values)
report("return/ER@0.01",  0.15 < er_001  < 0.40,
       f"{er_001:.3f} (paper: ~0.25)")
report("return/ER@0.025", 0.20 < er_0025 < 0.45,
       f"{er_0025:.3f} (paper: ~0.29)")
report("return/ER@0.975", 0.70 < er_0975 < 0.95,
       f"{er_0975:.3f} (paper: ~0.82)")
report("return/ER@0.99",  0.70 < er_099  < 0.97,
       f"{er_099:.3f} (paper: ~0.84)")

# ─────────────────────────────────────────────────────────────────────────────
# Check 5 — ATM 3-month IV distribution (Table 2)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 5: ATM 3-month IV exceedance ratios ──")

# Log-IV flat order is tau-major: flat[tau_idx * 10 + m_idx]
# condition_test last 80 cols = log-IV surface at current time
# true_test[:, 1:] = log-IV increments

# ATM = moneyness closest to 1.0; 3m = tau closest to 91/365
atm_m_idx = int(np.argmin(np.abs(MONEYNESS_GRID - 1.0)))
atm_t_idx = int(np.argmin(np.abs(TAU_GRID - 91/365)))
flat_idx   = atm_t_idx * len(MONEYNESS_GRID) + atm_m_idx  # tau-major

# Current log-IV at each test day (from conditioning vector, columns 3:)
cond_np = cond_test.numpy() if isinstance(cond_test, torch.Tensor) else cond_test
cur_log_iv = cond_np[:, 3:]                        # [N_test, 80]

# True next log-IV at the ATM 3m point
true_next_log_iv_atm = (cur_log_iv[:, flat_idx]
                        + true_test[:, 1 + flat_idx])
true_atm_iv = np.exp(true_next_log_iv_atm)

# Generated next log-IV — repeat cur_log_iv for K samples per day
cur_log_iv_rep = np.repeat(cur_log_iv, SAMPLES_PER_DAY, axis=0)  # [N_test*K, 80]
gen_next_log_iv_atm = cur_log_iv_rep[:, flat_idx] + gen_samples[:, 1 + flat_idx]
gen_atm_iv = np.exp(gen_next_log_iv_atm)

q_iv_001  = np.quantile(true_atm_iv, 0.010)
q_iv_0025 = np.quantile(true_atm_iv, 0.025)
q_iv_0975 = np.quantile(true_atm_iv, 0.975)

er_iv_001  = exceedance_ratio(gen_atm_iv, q_iv_001)
er_iv_0975 = exceedance_ratio(gen_atm_iv, q_iv_0975)

print(f"  ATM moneyness index: {atm_m_idx} ({MONEYNESS_GRID[atm_m_idx]:.3f}), "
      f"tau index: {atm_t_idx} ({TAU_DAYS[atm_t_idx]}d)")
print(f"  True ATM IV range: [{true_atm_iv.min():.3f}, {true_atm_iv.max():.3f}], "
      f"mean={true_atm_iv.mean():.3f}")

report("atm_iv/ER@0.01",  0.05 < er_iv_001  < 0.30,
       f"{er_iv_001:.3f} (paper 3m ATM: ~0.14)")
report("atm_iv/ER@0.975", 0.35 < er_iv_0975 < 0.65,
       f"{er_iv_0975:.3f} (paper 3m ATM: ~0.50)")

# ─────────────────────────────────────────────────────────────────────────────
# Check 6 — PCA alignment (Tables 4–5)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 6: PCA alignment ──")

true_iv_incr  = true_test[:, 1:]                      # [N_test, 80]
gen_iv_incr   = gen_samples[:, 1:]                    # [N_test*K, 80]

true_evals, true_pcs  = top_k_pcs(true_iv_incr, k=3)
gen_evals,  gen_pcs   = top_k_pcs(gen_iv_incr,  k=3)

all_evals_true = np.linalg.eigvalsh(
    (true_iv_incr - true_iv_incr.mean(0)).T @
    (true_iv_incr - true_iv_incr.mean(0)) / len(true_iv_incr)
)[::-1]

var_exp_gen = variance_explained(gen_evals)

report("pca/pc1_var_exp", 0.43 < var_exp_gen[0] < 0.48,
       f"{var_exp_gen[0]:.3f} (paper: 0.4531 ± 0.0184)")

for k, (thresh, name) in enumerate([(0.90, "PC1"), (0.90, "PC2"), (0.75, "PC3")]):
    inner = abs(float(true_pcs[:, k] @ gen_pcs[:, k]))
    report(f"pca/{name}_align", inner > thresh,
           f"|PC{k+1}_true · PC{k+1}_gen| = {inner:.3f} (expected > {thresh})")

# ─────────────────────────────────────────────────────────────────────────────
# Check 7 — Leverage effect (Tables 6–7)
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 7: Leverage-effect correlation ──")

# PC1 scores of generated IV increments
gen_iv_centered = gen_iv_incr - gen_iv_incr.mean(0)
pc1_scores = gen_iv_centered @ gen_pcs[:, 0]           # [N_test*K]
leverage_corr = float(np.corrcoef(gen_ret, pc1_scores)[0, 1])

report("leverage/corr_ret_pc1", -0.84 < leverage_corr < -0.68,
       f"corr(ΔlogS, PC1) = {leverage_corr:.3f} (expected −0.84 to −0.68)")

# ─────────────────────────────────────────────────────────────────────────────
# Check 8 — Visual vol-surface plot
# ─────────────────────────────────────────────────────────────────────────────

print("\n── Check 8: Vol surface plot ──")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reconstruct 3 random sample surfaces from first test-day conditioning
rng = np.random.default_rng(0)
sample_idx = rng.choice(SAMPLES_PER_DAY, size=3, replace=False)
first_cur_log_iv = cond_np[0, 3:]                     # [80]

fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                          subplot_kw={"projection": "3d"})

for ax, i in zip(axes, sample_idx):
    log_iv_next = first_cur_log_iv + gen_samples[i, 1:]   # [80]
    iv_surface = np.exp(log_iv_next).reshape(len(TAU_DAYS), len(MONEYNESS_GRID))
    # iv_surface: [tau, moneyness] → plot with axes swapped for typical display
    M, T = np.meshgrid(MONEYNESS_GRID, TAU_DAYS)
    ax.plot_surface(M, T, iv_surface, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Tau (days)")
    ax.set_zlabel("IV")
    ax.set_title(f"Sample {i}")

fig.suptitle("Generated vol surfaces — sanity check (first test day conditioning)")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=100)
plt.close(fig)
print(f"  Saved → {PLOT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
passes = sum(1 for _, ok, _ in _results if ok)
total  = len(_results)
for name, ok, detail in _results:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")

print(f"\n{passes}/{total} checks passed")
if passes == total:
    print("Checkpoint matches expected VolGAN architecture and statistics.")
else:
    print("Some checks failed — review FAIL lines above.")
