"""Leave-one-task-out transfer: train on all tasks but one, test on the held-out task.

Separate from finetune_transfer.py (which it imports but does not change).

For each held-out task T and each seed:
  - fresh base model + fresh LoRA adapter
  - train for --steps steps on the other tasks, with each task contributing
    equally often (tasks are balanced, not weighted by how many items they have)
  - score the held-out task T

Same training settings as finetune_transfer.py, so results are comparable with
the single-task transfer cells.

Output (in --out):
  base_scores.csv                 base model on every task
  loo_<T>_seed<s>.csv             fine-tuned scores on held-out task T
Finished files are skipped, so runs can be stopped and resumed.

Example:
    python src/leave_one_out.py --model allenai/OLMo-2-1124-7B --data_dir data/raw \
        --out results_loo/OLMo-2-1124-7B --seeds 0 1 2 \
        --tasks FB1 FB2 IR FB_attn --micro_batch 1
"""

import argparse
import math
import os
from functools import reduce

import pandas as pd

from data import load_tasks
from finetune_transfer import DTYPES, finetune, free_memory, load_model, score


def balanced_train_set(tasks, names):
    """Combine tasks so each appears equally often when sampling uniformly.

    Each task's rows are repeated so every task has the same number of rows
    (the least common multiple of the task sizes). E.g. with FB1 (192),
    FB2 (64) and FB_attn (384): FB1 x2, FB2 x6, FB_attn x1 -> 384 rows each.
    """
    sizes = [len(tasks[n]) for n in names]
    target = reduce(lambda a, b: a * b // math.gcd(a, b), sizes)
    parts = [pd.concat([tasks[n]] * (target // len(tasks[n])), ignore_index=True)
             for n in names]
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=["FB1", "FB2", "IR", "FB_attn"])
    ap.add_argument("--holdout", nargs="+", default=None,
                    help="Which tasks to hold out (default: each task in turn).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    # Same defaults as finetune_transfer.py
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tasks = load_tasks(args.data_dir, args.tasks)
    holdouts = args.holdout or args.tasks

    # 1. Base model scores (no fine-tuning).
    base_path = os.path.join(args.out, "base_scores.csv")
    if not os.path.exists(base_path):
        model, tok = load_model(args.model, DTYPES[args.dtype])
        base = pd.concat([score(model, tok, df, f"base {n}") for n, df in tasks.items()])
        base["model"] = args.model
        base.to_csv(base_path, index=False)
        del model
        free_memory()

    # 2. Leave-one-out runs.
    for seed in args.seeds:
        for held_out in holdouts:
            path = os.path.join(args.out, f"loo_{held_out}_seed{seed}.csv")
            if os.path.exists(path):
                print(f"skip {path}")
                continue
            train_names = [n for n in args.tasks if n != held_out]
            source = "+".join(train_names)
            print(f"seed {seed} | train {source} | test {held_out}")

            train_df = balanced_train_set(tasks, train_names)
            model, tok = finetune(args, train_df, seed)
            out = score(model, tok, tasks[held_out], f"{source}->{held_out}")
            del model
            free_memory()

            out["source"] = source
            out["held_out"] = held_out
            out["seed"] = seed
            out["model"] = args.model
            out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
