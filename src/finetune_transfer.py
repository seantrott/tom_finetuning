"""Cross-task transfer between EPITOME tasks via LoRA fine-tuning.

For each source task S and each seed:
  - Cross-task: fine-tune on all of S, then score every other target task T.
  - Within-task (the diagonal): k-fold CV over S's item groups; fine-tune on
    k-1 folds, score the held-out fold.
The base model (no fine-tuning) is scored once on every task.

Every source task gets the same number of gradient steps, so tasks with more
items (FB1 has 192) don't get more training than tasks with fewer.

Output: one CSV row per (seed, source, target, item) with the score
log P(correct) - log P(incorrect) after fine-tuning. Base scores are in
base_scores.csv. make_matrix.py turns these into the transfer matrix.

Example:
    python finetune_transfer.py --model allenai/OLMo-2-0425-1B \
        --data_dir data/raw --out results/olmo2-1b \
        --tasks FB1 FB2 IR RM FB_attn --seeds 0 1 2
"""

import argparse
import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_tasks


# ---------------------------------------------------------------- utilities

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def free_memory():
    """Release GPU memory between runs (NVIDIA or Apple Silicon)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_model(name, dtype):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="auto")
    return model, tok


def encode(tok, prompt, completion):
    """Tokenize prompt and completion separately, then concatenate.

    Doing it this way makes the prompt/completion boundary identical for
    training and scoring. Returns (input_ids, number of completion tokens).
    """
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    return p + c, len(c)


# ---------------------------------------------------------------- scoring

@torch.no_grad()
def completion_logprob(model, tok, prompt, completion):
    ids, n_c = encode(tok, prompt, completion)
    x = torch.tensor([ids], device=model.device)
    logprobs = torch.log_softmax(model(x).logits[0].float(), dim=-1)
    # Token at position i is predicted by logits at position i-1.
    targets = x[0, -n_c:]
    preds = logprobs[-n_c - 1:-1]
    return preds.gather(1, targets.unsqueeze(1)).sum().item()


def score(model, tok, df, desc="scoring"):
    model.eval()
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        lp_c = completion_logprob(model, tok, r["prompt"], r["correct"])
        lp_i = completion_logprob(model, tok, r["prompt"], r["incorrect"])
        rows.append({
            "target": r["task"], "group": r["group"], "item_id": r["item_id"],
            "condition": r["condition"], "lp_correct": lp_c, "lp_incorrect": lp_i,
            "log_odds": lp_c - lp_i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- training

def make_batch(tok, examples, device):
    """Right-padded batch; loss only on the completion tokens."""
    encoded = [encode(tok, r["prompt"], r["correct"]) for r in examples]
    max_len = max(len(ids) for ids, _ in encoded)
    input_ids, labels, attn = [], [], []
    for ids, n_c in encoded:
        pad = max_len - len(ids)
        input_ids.append(ids + [tok.pad_token_id] * pad)
        labels.append([-100] * (len(ids) - n_c) + ids[-n_c:] + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    t = lambda a: torch.tensor(a, device=device)
    return t(input_ids), t(labels), t(attn)


def finetune(args, train_df, seed):
    """Fresh base model + fresh LoRA adapter, trained for a fixed number of steps."""
    set_seed(seed)
    model, tok = load_model(args.model, DTYPES[args.dtype])
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    records = train_df.to_dict("records")
    rng = random.Random(seed)
    for _ in tqdm(range(args.steps), desc="training", leave=False):
        batch = rng.sample(records, min(args.batch_size, len(records)))
        # Process the batch in small pieces and accumulate gradients, so long
        # passages (e.g. RM stories) don't run out of memory. One optimizer
        # step per full batch, same as before.
        chunks = [batch[i:i + args.micro_batch] for i in range(0, len(batch), args.micro_batch)]
        for chunk in chunks:
            input_ids, labels, attn = make_batch(tok, chunk, model.device)
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            (loss * len(chunk) / len(batch)).backward()
            del input_ids, labels, attn, loss
        opt.step()
        opt.zero_grad()
    del opt
    return model, tok


def group_folds(df, k, seed):
    """Split item groups (not rows) into k folds."""
    groups = sorted(df["group"].unique())
    random.Random(seed).shuffle(groups)
    return [set(groups[i::k]) for i in range(k)]


# ---------------------------------------------------------------- main

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=["FB1", "FB2", "IR", "RM", "FB_attn"],
                    help="Tasks used as both sources and targets.")
    ap.add_argument("--custom", nargs="*", default=[],
                    help="Extra tasks as NAME=path.csv (e.g. a matched control).")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=4, help="k for the within-task diagonal")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--micro_batch", type=int, default=2,
                    help="Examples processed at once; lower this if you run out of memory")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    custom = dict(c.split("=", 1) for c in args.custom)
    names = args.tasks + [n for n in custom if n not in args.tasks]
    tasks = load_tasks(args.data_dir, names, custom)

    # 1. Base model scores (no fine-tuning).
    base_path = os.path.join(args.out, "base_scores.csv")
    if not os.path.exists(base_path):
        model, tok = load_model(args.model, DTYPES[args.dtype])
        base = pd.concat([score(model, tok, df, f"base {n}") for n, df in tasks.items()])
        base["model"] = args.model
        base.to_csv(base_path, index=False)
        del model
        free_memory()

    # 2. Fine-tuned scores. One file per (seed, source) so runs can resume.
    for seed in args.seeds:
        for source, src_df in tasks.items():
            path = os.path.join(args.out, f"ft_{source}_seed{seed}.csv")
            if os.path.exists(path):
                print(f"skip {path}")
                continue
            print(f"seed {seed} | source {source}")
            results = []

            # Cross-task: train on all of the source, test on every other task.
            model, tok = finetune(args, src_df, seed)
            for target, tgt_df in tasks.items():
                if target != source:
                    results.append(score(model, tok, tgt_df, f"{source}->{target}"))
            del model
            free_memory()

            # Within-task: train on k-1 folds of item groups, test on the held-out fold.
            for fold_i, held_out in enumerate(group_folds(src_df, args.folds, seed)):
                train = src_df[~src_df["group"].isin(held_out)]
                test = src_df[src_df["group"].isin(held_out)]
                model, tok = finetune(args, train, seed)
                s = score(model, tok, test, f"{source} fold {fold_i}")
                s["fold"] = fold_i
                results.append(s)
                del model
                free_memory()

            out = pd.concat(results)
            out["source"] = source
            out["seed"] = seed
            out["model"] = args.model
            out.to_csv(path, index=False)


if __name__ == "__main__":
    main()