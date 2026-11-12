"""Build the transfer matrix (train task x test task) and plot it.

Each cell = mean improvement in log-odds of the correct answer on the test task
after fine-tuning on the train task, relative to the base model:

    gain(S -> T) = mean over T's items of [log_odds_after - log_odds_base]

If --control is given, the control's gain on T is subtracted:

    transfer(S -> T) = gain(S -> T) - gain(control -> T)

so 0 means "no better than fine-tuning on the control task".

Example:
    python make_matrix.py --results results/olmo2-1b --control FB_attn \
        --order FB1 FB2 IR RM
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(results_dir):
    base = pd.read_csv(os.path.join(results_dir, "base_scores.csv"))
    ft = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(results_dir, "ft_*.csv")))
    base = base[["target", "item_id", "log_odds"]].rename(columns={"log_odds": "log_odds_base"})
    df = ft.merge(base, on=["target", "item_id"], how="left")
    df["gain"] = df["log_odds"] - df["log_odds_base"]
    return df


def transfer_table(df, control=None):
    """Per (seed, source, target) mean gain, minus the control's gain if given."""
    g = df.groupby(["seed", "source", "target"], as_index=False)["gain"].mean()
    if control is not None:
        ctrl = g[g["source"] == control][["seed", "target", "gain"]].rename(
            columns={"gain": "control_gain"})
        g = g.merge(ctrl, on=["seed", "target"], how="left")
        g["transfer"] = g["gain"] - g["control_gain"]
        g = g[(g["source"] != control) & (g["target"] != control)]
    else:
        g["transfer"] = g["gain"]
    return g


def plot(matrix, path, title):
    tasks = list(matrix.index)
    vmax = np.nanmax(np.abs(matrix.values))
    fig, ax = plt.subplots(figsize=(1.2 * len(tasks) + 2, 1.1 * len(tasks) + 1))
    im = ax.imshow(matrix.values, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_yticks(range(len(tasks)), tasks)
    ax.set_xlabel("Test task")
    ax.set_ylabel("Train task")
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            v = matrix.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 * vmax else "black")
        # Outline the within-task (held-out items) diagonal.
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.5))
    fig.colorbar(im, ax=ax, label="Δ log-odds correct")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--control", default=None, help="Source task to subtract, e.g. FB_attn")
    ap.add_argument("--order", nargs="+", default=None, help="Row/column order")
    args = ap.parse_args()

    df = load(args.results)
    t = transfer_table(df, args.control)

    summary = t.groupby(["source", "target"])["transfer"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(os.path.join(args.results, "transfer_matrix_long.csv"), index=False)

    matrix = summary.pivot(index="source", columns="target", values="mean")
    order = args.order or sorted(matrix.index)
    matrix = matrix.reindex(index=order, columns=order)
    matrix.to_csv(os.path.join(args.results, "transfer_matrix.csv"))

    title = "Transfer vs. " + args.control if args.control else "Gain vs. base model"
    plot(matrix, os.path.join(args.results, "transfer_matrix.png"), title)
    print(matrix.round(2))


if __name__ == "__main__":
    main()
