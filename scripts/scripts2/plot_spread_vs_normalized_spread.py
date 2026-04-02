#!/usr/bin/env python3
"""Plot raw spread vs normalized-advantage spread for different normalizers.

This helps visualize how strongly each normalization method equalizes advantage
magnitudes across groups with different intrinsic spread.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_tau_values(value: str) -> list[float]:
    """Parse comma-separated tau values from CLI."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("--tau-values must contain at least one numeric value")
    taus = [float(p) for p in parts]
    if any(t <= 0 for t in taus):
        raise ValueError("All tau values must be > 0")
    return taus


def _make_group_with_std(group_size: int, target_std: float, rng: np.random.Generator) -> np.ndarray:
    """Sample a zero-mean group and rescale it to the requested std."""
    x = rng.standard_normal(group_size)
    x = x - x.mean()
    std = x.std()
    if std < 1e-12:
        x = np.linspace(-1.0, 1.0, num=group_size)
        x = x - x.mean()
        std = x.std()
    return x * (target_std / std)


def _apply_normalization(x: np.ndarray, method: str, eps: float, tau: float, alpha: float) -> np.ndarray:
    """Apply one normalization rule to centered group values."""
    std = x.std()
    if method == "std":
        denom = std + eps
    elif method == "floor":
        denom = max(std, tau)
    elif method == "additive":
        denom = std + tau
    elif method == "power":
        denom = (std + eps) ** alpha
    else:
        raise ValueError(f"Unknown method: {method}")
    return x / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real spread vs normalized spread.")
    parser.add_argument("--group-size", type=int, default=8, help="Samples per group.")
    parser.add_argument("--trials", type=int, default=300, help="Trials per raw spread point.")
    parser.add_argument("--num-points", type=int, default=48, help="Number of raw spread points.")
    parser.add_argument("--min-spread", type=float, default=1e-5, help="Minimum raw std to test.")
    parser.add_argument("--max-spread", type=float, default=1.0, help="Maximum raw std to test.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon used in std normalizers.")
    parser.add_argument("--tau", type=float, default=5e-2, help="Floor/additive constant.")
    parser.add_argument(
        "--tau-values",
        type=str,
        default="",
        help="Comma-separated tau sweep values (e.g. '0.005,0.02,0.05,0.1'). If empty, uses --tau.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Exponent for partial normalization.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/spread_vs_normalized_spread.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_spreads = np.logspace(np.log10(args.min_spread), np.log10(args.max_spread), args.num_points)
    taus = _parse_tau_values(args.tau_values) if args.tau_values else [args.tau]

    # Always compute std and power references once.
    ref_methods = ["std", "power"]
    ref_curves: dict[str, list[float]] = {m: [] for m in ref_methods}
    floor_curves: dict[float, list[float]] = {tau: [] for tau in taus}
    additive_curves: dict[float, list[float]] = {tau: [] for tau in taus}

    for target_std in raw_spreads:
        ref_spreads: dict[str, list[float]] = {m: [] for m in ref_methods}
        floor_spreads: dict[float, list[float]] = {tau: [] for tau in taus}
        additive_spreads: dict[float, list[float]] = {tau: [] for tau in taus}

        for _ in range(args.trials):
            centered = _make_group_with_std(args.group_size, target_std, rng)

            for m in ref_methods:
                normalized = _apply_normalization(centered, m, args.eps, args.tau, args.alpha)
                ref_spreads[m].append(float(normalized.std()))

            for tau in taus:
                floor_norm = _apply_normalization(centered, "floor", args.eps, tau, args.alpha)
                add_norm = _apply_normalization(centered, "additive", args.eps, tau, args.alpha)
                floor_spreads[tau].append(float(floor_norm.std()))
                additive_spreads[tau].append(float(add_norm.std()))

        for m in ref_methods:
            ref_curves[m].append(float(np.mean(ref_spreads[m])))
        for tau in taus:
            floor_curves[tau].append(float(np.mean(floor_spreads[tau])))
            additive_curves[tau].append(float(np.mean(additive_spreads[tau])))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)

    # Left: floor sweep
    ax = axes[0]
    ax.plot(raw_spreads, ref_curves["std"], linestyle="--", linewidth=2, color="black", label="std reference")
    for tau in taus:
        ax.plot(raw_spreads, floor_curves[tau], linewidth=2, label=f"tau={tau:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Floor Normalization: x/max(sigma, tau)")
    ax.set_xlabel("Real spread (raw centered std)")
    ax.set_ylabel("Normalized advantage spread (std)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Right: additive sweep
    ax = axes[1]
    ax.plot(raw_spreads, ref_curves["std"], linestyle="--", linewidth=2, color="black", label="std reference")
    for tau in taus:
        ax.plot(raw_spreads, additive_curves[tau], linewidth=2, label=f"tau={tau:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Additive Normalization: x/(sigma + tau)")
    ax.set_xlabel("Real spread (raw centered std)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Tau Sweep: Real Spread vs Normalized Spread", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(args.output, dpi=160)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()