#!/usr/bin/env python3
"""Plot fixed and adaptive lambda trace shapes over token positions."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def token_grid(length: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(length, dtype=np.float64)
    if length <= 1:
        u = np.ones_like(t)
    else:
        u = t / (length - 1.0)
    return t, u


def fixed_trace(t: np.ndarray, length: int, lam: float, gamma: float = 1.0) -> np.ndarray:
    return (gamma * lam) ** ((length - 1.0) - t)


def start_credit(length: int, t_min: int, alpha: float, p: float, a_min: float) -> float:
    return float(np.clip(alpha * (t_min / length) ** p, a_min, 1.0))


def shared_shortest_exp_trace(
    t: np.ndarray,
    length: int,
    t_min: int,
    alpha: float,
    gamma: float,
    lam_min: float,
    lam_max: float,
) -> tuple[np.ndarray, float]:
    # One base shared by the whole group, calibrated from shortest rollout.
    # Longer rollouts then extend the same exponential trace further to the left.
    lam_eff = float(np.clip(alpha ** (1.0 / max(t_min - 1, 1)) / gamma, lam_min, lam_max))
    w = fixed_trace(t, length, lam_eff, gamma)
    return w, lam_eff


def linear_trace(u: np.ndarray, length: int, t_min: int, alpha: float, p: float, a_min: float) -> tuple[np.ndarray, float]:
    a = start_credit(length, t_min, alpha, p, a_min)
    w = a + (1.0 - a) * u
    return w, a


def power_trace(
    u: np.ndarray,
    length: int,
    t_min: int,
    alpha: float,
    p: float,
    beta: float,
    a_min: float,
) -> tuple[np.ndarray, float, float]:
    a = start_credit(length, t_min, alpha, p, a_min)
    eta = 1.0 + beta * (1.0 - t_min / length)
    w = a + (1.0 - a) * (u ** eta)
    return w, a, eta


def logistic_trace(
    u: np.ndarray,
    length: int,
    t_min: int,
    alpha: float,
    p: float,
    kappa_0: float,
    kappa_1: float,
    a_min: float,
) -> tuple[np.ndarray, float, float]:
    a = start_credit(length, t_min, alpha, p, a_min)
    kappa = kappa_0 + kappa_1 * (1.0 - t_min / length)

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    num = sigmoid(kappa * (u - 0.5)) - sigmoid(np.array([-kappa / 2.0]))[0]
    den = sigmoid(np.array([kappa / 2.0]))[0] - sigmoid(np.array([-kappa / 2.0]))[0]
    s = num / max(den, 1e-12)
    w = a + (1.0 - a) * s
    return w, a, kappa


def mass_normalize(w: np.ndarray, length: int) -> np.ndarray:
    return w * (length / (np.sum(w) + 1e-12))


def main() -> None:
    lengths = [620, 680, 740, 790]
    t_min = min(lengths)

    alpha = 0.25
    gamma = 1.0

    fixed_lambdas = [0.98, 0.99, 0.995, 0.999]

    p = 2.0
    a_min = 0.05
    beta = 3.0
    kappa_0 = 3.0
    kappa_1 = 6.0
    lam_min = 0.985
    lam_max = 0.9997

    mass_norm = False

    out_dir = Path("reports/figures/lambda_trace_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Fixed lambdas at T=790
    t_ref_len = 790
    t_ref, u_ref = token_grid(t_ref_len)
    fig, ax = plt.subplots(figsize=(9, 5))
    for lam in fixed_lambdas:
        w = fixed_trace(t_ref, t_ref_len, lam, gamma)
        ax.plot(u_ref, w, linewidth=2, label=f"fixed lambda={lam}")
    ax.set_title(f"Fixed Lambda Traces (T={t_ref_len})")
    ax.set_xlabel("Normalized token position t/(T-1)")
    ax.set_ylabel("Trace weight")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fixed_lambdas_T790.png", dpi=220)
    plt.close(fig)

    # Plot 2: Adaptive variants at T=790
    w_exp, lam_eff = shared_shortest_exp_trace(
        t_ref,
        t_ref_len,
        t_min,
        alpha,
        gamma,
        lam_min,
        lam_max,
    )
    w_lin, a_lin = linear_trace(u_ref, t_ref_len, t_min, alpha, p, a_min)
    w_pow, a_pow, eta = power_trace(u_ref, t_ref_len, t_min, alpha, p, beta, a_min)
    w_log, a_log, kappa = logistic_trace(u_ref, t_ref_len, t_min, alpha, p, kappa_0, kappa_1, a_min)

    if mass_norm:
        w_exp = mass_normalize(w_exp, t_ref_len)
        w_lin = mass_normalize(w_lin, t_ref_len)
        w_pow = mass_normalize(w_pow, t_ref_len)
        w_log = mass_normalize(w_log, t_ref_len)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(u_ref, w_exp, linewidth=2.2, label=f"shared-shortest-exp (lambda_eff={lam_eff:.5f})")
    ax.plot(u_ref, w_lin, linewidth=2.2, label=f"linear (a={a_lin:.3f})")
    ax.plot(u_ref, w_pow, linewidth=2.2, label=f"power (a={a_pow:.3f}, eta={eta:.3f})")
    ax.plot(u_ref, w_log, linewidth=2.2, label=f"logistic (a={a_log:.3f}, kappa={kappa:.3f})")
    ax.set_title(f"Adaptive Trace Variants (T={t_ref_len}, T_min={t_min}, alpha={alpha})")
    ax.set_xlabel("Normalized token position t/(T-1)")
    ax.set_ylabel("Trace weight")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "adaptive_variants_T790.png", dpi=220)
    plt.close(fig)

    # Plot 3: First-token credit vs length
    t_range = np.arange(t_min, 1201, 10)
    a_vals = np.array([start_credit(t, t_min, alpha, p, a_min) for t in t_range])
    w0_fixed_099 = np.array([0.99 ** (t - 1) for t in t_range])
    w0_fixed_0995 = np.array([0.995 ** (t - 1) for t in t_range])

    lam_eff_global = float(np.clip(alpha ** (1.0 / max(t_min - 1, 1)) / gamma, lam_min, lam_max))
    w0_adexp = np.array([(gamma * lam_eff_global) ** (t - 1) for t in t_range])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_range, w0_fixed_099, linewidth=2, label="fixed lambda=0.99")
    ax.plot(t_range, w0_fixed_0995, linewidth=2, label="fixed lambda=0.995")
    ax.plot(t_range, w0_adexp, linewidth=2, label=f"shared-shortest-exp (lambda_eff={lam_eff_global:.5f})")
    ax.plot(t_range, a_vals, linewidth=2, label=f"start credit a(T), p={p}")
    ax.set_title("First-Token Credit vs Sequence Length")
    ax.set_xlabel("Sequence length T")
    ax.set_ylabel("First-token trace weight")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "first_token_credit_vs_length.png", dpi=220)
    plt.close(fig)

    # Plot 4: Adaptive variants by length (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, length in zip(axes.flat, lengths):
        t, u = token_grid(length)
        w_exp, _ = shared_shortest_exp_trace(t, length, t_min, alpha, gamma, lam_min, lam_max)
        w_lin, _ = linear_trace(u, length, t_min, alpha, p, a_min)
        w_pow, _, _ = power_trace(u, length, t_min, alpha, p, beta, a_min)
        w_log, _, _ = logistic_trace(u, length, t_min, alpha, p, kappa_0, kappa_1, a_min)

        if mass_norm:
            w_exp = mass_normalize(w_exp, length)
            w_lin = mass_normalize(w_lin, length)
            w_pow = mass_normalize(w_pow, length)
            w_log = mass_normalize(w_log, length)

        ax.plot(u, w_exp, linewidth=1.9, label="shared-shortest-exp")
        ax.plot(u, w_lin, linewidth=1.9, label="linear")
        ax.plot(u, w_pow, linewidth=1.9, label="power")
        ax.plot(u, w_log, linewidth=1.9, label="logistic")
        ax.set_title(f"T={length}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adaptive Lambda Trace Shapes Across Lengths", y=0.98)
    fig.text(0.5, 0.02, "Normalized token position t/(T-1)", ha="center")
    fig.text(0.02, 0.5, "Trace weight", va="center", rotation="vertical")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=[0.03, 0.04, 1.0, 0.90])
    fig.savefig(out_dir / "adaptive_variants_by_length.png", dpi=220)
    plt.close(fig)

    # Plot 5: Group-shared shortest-anchored trace aligned to end token.
    # If your desired behavior is implemented, all curves overlap on the suffix
    # of length T_min and only longer samples extend further left.
    fig, ax = plt.subplots(figsize=(9, 5))
    for length in lengths:
        t, _ = token_grid(length)
        w_shared, lam_eff_local = shared_shortest_exp_trace(t, length, t_min, alpha, gamma, lam_min, lam_max)
        x = t - (length - 1)  # 0 at last token; negative means earlier tokens
        ax.plot(x, w_shared, linewidth=2.1, label=f"T={length}")

    ax.set_title(
        "Group-Shared Shortest-Anchored Trace (Aligned to End)\n"
        f"lambda_eff={lam_eff_local:.5f}, T_min={t_min}, alpha={alpha}"
    )
    ax.set_xlabel("Token offset from end (0 = last token)")
    ax.set_ylabel("Trace weight")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "shared_shortest_trace_aligned_end.png", dpi=220)
    plt.close(fig)

    print("Saved plots:")
    for path in sorted(out_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
