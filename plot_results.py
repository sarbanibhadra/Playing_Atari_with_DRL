"""
Plot the two training curves used in Figure 2 of the paper:
average reward per episode and average max Q on held-out states.

Usage:
  python plot_results.py --run-dir runs/pong
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/pong")
    p.add_argument("--window", type=int, default=20, help="moving-average window for training returns")
    args = p.parse_args()

    train = read_csv(os.path.join(args.run_dir, "train_episodes.csv"))
    evals = read_csv(os.path.join(args.run_dir, "eval.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    if train:
        ax.plot(train["step"], train["return"], alpha=0.3, label="episode return")
        if len(train["return"]) >= args.window:
            kernel = np.ones(args.window) / args.window
            smooth = np.convolve(train["return"], kernel, mode="valid")
            ax.plot(train["step"][args.window - 1:], smooth, label=f"moving avg ({args.window})")
    ax.set_title("Training episode return")
    ax.set_xlabel("agent steps")
    ax.set_ylabel("return")
    ax.legend()

    ax = axes[1]
    if evals:
        ax.plot(evals["step"], evals["eval_return"], marker="o")
    ax.set_title("Evaluation return (epsilon = 0.05)")
    ax.set_xlabel("agent steps")
    ax.set_ylabel("average return per episode")

    ax = axes[2]
    if evals:
        ax.plot(evals["step"], evals["avg_max_q"], marker="o", color="tab:green")
    ax.set_title("Average max Q on held-out states")
    ax.set_xlabel("agent steps")
    ax.set_ylabel("average action value (Q)")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(args.run_dir, "training_curves.png")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
