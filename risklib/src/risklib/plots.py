"""Minimal plotting helpers that save straight to a PNG file."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)   # create figures/ if missing
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_series(y, path, title="", ylabel="", shade_from=None):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(np.arange(1, len(y) + 1), y, lw=0.8)
    if shade_from is not None:
        ax.axvspan(shade_from, len(y), color="tab:red", alpha=0.12)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title=title, xlabel="day", ylabel=ylabel)
    _save(fig, path)


def plot_hists(data: dict, path, bins=120, log=True, xlabel=""):
    fig, axes = plt.subplots(1, len(data), figsize=(5.5 * len(data), 4))
    for ax, (name, x) in zip(np.atleast_1d(axes), data.items()):
        ax.hist(x, bins=bins, color="steelblue")
        if log:
            ax.set_yscale("log")
        ax.set_title(name.replace("$", r"\$"))
        ax.set_xlabel(xlabel)
    _save(fig, path)


def plot_pairs(U, names, path, q=0.025):
    U = np.asarray(U)
    pairs = [(i, j) for i in range(U.shape[1]) for j in range(i + 1, U.shape[1])]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.3 * len(pairs), 4.2))
    for ax, (i, j) in zip(np.atleast_1d(axes), pairs):
        ax.scatter(U[:, i], U[:, j], s=4, alpha=0.5)
        for v in (q, 1 - q):
            ax.axvline(v, color="r", lw=0.6)
            ax.axhline(v, color="r", lw=0.6)
        ax.set(xlabel=names[i], ylabel=names[j], aspect="equal")
    _save(fig, path)


def plot_bars(y, path, title="", ylabel="", highlight=None):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    color = "tab:blue" if highlight is None else np.where(highlight, "tab:red", "tab:blue")
    ax.bar(np.arange(len(y)), y, width=1.0, color=color)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(title=title, xlabel="day", ylabel=ylabel)
    _save(fig, path)
