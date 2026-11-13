"""Convert each EPITOME task into one shared format.

Every task becomes a DataFrame with these columns:
    task       task name (e.g. "FB1")
    group      id shared by all variants of the same story/item; used so that
               variants of one item never end up on both sides of a train/test split
    item_id    unique id for the row
    condition  the task's manipulation (e.g. "False Belief", "unaware")
    prompt     text the model sees
    correct    correct continuation (what we train on and score)
    incorrect  the competing continuation (only used for scoring)

Scoring is always: log P(correct | prompt) - log P(incorrect | prompt).
"""

import os
import pandas as pd

COLUMNS = ["task", "group", "item_id", "condition", "prompt", "correct", "incorrect"]


def load_fb1(data_dir):
    """Location-based false belief (192 rows = 12 items x 16 variants).

    Prompt is the passage up to "[MASK]"; the two continuations are the start
    and end locations. critical_a holds the correct one (start for False
    Belief, end for True Belief).
    """
    df = pd.read_csv(os.path.join(data_dir, "fb.csv"))
    out = pd.DataFrame({
        "task": "FB1",
        "group": "fb1_" + df["item"].astype(str),
        "item_id": df["item_id"],
        "condition": df["condition"],
        "prompt": df["passage"].str.replace(" [MASK].", "", regex=False).str.strip(),
        "correct": " " + df["critical_a"],
        "incorrect": " " + df.apply(
            lambda r: r["end"] if r["critical_a"] == r["start"] else r["start"], axis=1
        ),
    })
    return out[COLUMNS]


def _yes_no(df, task, prompt):
    """Shared helper for the Yes/No tasks (FB2, IR)."""
    correct = df["critical_a"].str.strip()
    return pd.DataFrame({
        "task": task,
        "group": task.lower() + "_" + df["item"].astype(str),
        "item_id": (task.lower() + "_" + df["item"].astype(str) + "_"
                    + df["speaker_knowledge"] + "_" + df["knowledge_cue"]),
        "condition": df["speaker_knowledge"],
        "prompt": prompt,
        "correct": " " + correct,
        "incorrect": " " + correct.map({"Yes": "No", "No": "Yes"}),
    })[COLUMNS]


def load_fb2(data_dir):
    """Knowledge-state false belief (64 rows = 16 items x 4 variants), Yes/No."""
    df = pd.read_csv(os.path.join(data_dir, "fb2.csv"), encoding="utf-8-sig")
    prompt = df["passage"].str.strip() + "\n\n" + df["critical_q"] + "\n\nAnswer:"
    return _yes_no(df, "FB2", prompt)


def load_ir(data_dir):
    """Indirect requests (64 rows = 16 items x 4 variants), Yes/No."""
    df = pd.read_csv(os.path.join(data_dir, "ir.csv"), encoding="utf-8-sig")
    df = df.dropna(subset=["critical_utterance"]).reset_index(drop=True)
    prompt = (df["passage"].str.strip() + "\n\n" + df["critical_utterance"]
              + "\n\n" + df["critical_q"] + "\n\nAnswer:")
    return _yes_no(df, "IR", prompt)




def load_fb_attn(data_dir):
    """FB attention checks.

    As in the original attention-check script, the four questions for each
    story are asked about every passage variant of that story in fb.csv
    (False Belief and True Belief, all orders). The explicit and implicit
    variants share the same passage, so duplicates are dropped:
    12 stories x 8 passages x 4 questions = 384 rows.

    Each question's distractor is the answer to its paired question (1<->2, 3<->4).
    item_id encodes the passage variant, e.g. fbattn_1_fb_1_s_e_q1, so the
    passage's belief condition ("_fb_" or "_tb_") can be recovered later.
    """
    qs = pd.read_csv(os.path.join(data_dir, "fb_attn_check.csv"), encoding="utf-8-sig")
    stims = pd.read_csv(os.path.join(data_dir, "fb.csv"), encoding="utf-8-sig")
    # Variant id without the _ex/_im suffix, e.g. "1_fb_1_s_e"; keep one
    # passage per variant (the explicit version where the two differ slightly).
    stims["variant"] = stims["item_id"].str.replace(r"_(ex|im)$", "", regex=True)
    stims = (stims.sort_values("item_id")
                  .drop_duplicates(subset=["item", "variant"])
                  .copy())

    q_cols = [c for c in qs.columns if c.startswith("attn_check_")]
    df = stims[["item", "variant", "passage_hr"]].merge(qs[["item"] + q_cols], on="item")

    pairs = {1: 2, 2: 1, 3: 4, 4: 3}
    rows = []
    for _, r in df.iterrows():
        passage = str(r["passage_hr"]).strip()
        for q, other in pairs.items():
            rows.append({
                "task": "FB_attn",
                "group": f"fbattn_{r['item']}",
                "item_id": f"fbattn_{r['variant']}_q{q}",
                "condition": f"q{q}",
                "prompt": passage + " " + str(r[f"attn_check_{q}_q"]).strip(),
                "correct": " " + str(r[f"attn_check_{q}_a"]).strip(),
                "incorrect": " " + str(r[f"attn_check_{other}_a"]).strip(),
            })
    return pd.DataFrame(rows)[COLUMNS]


def load_custom(path, task_name):
    """Any extra task (e.g. a better-matched control) saved as a CSV with
    columns: group, item_id, condition, prompt, correct, incorrect."""
    df = pd.read_csv(path)
    df["task"] = task_name
    return df[COLUMNS]


LOADERS = {
    "FB1": load_fb1,
    "FB2": load_fb2,
    "IR": load_ir,
    "FB_attn": load_fb_attn,
}


def load_tasks(data_dir, names, custom=None):
    """Return {task_name: DataFrame}. `custom` maps extra task names to CSV paths."""
    custom = custom or {}
    tasks = {}
    for name in names:
        if name in custom:
            tasks[name] = load_custom(custom[name], name)
        else:
            tasks[name] = LOADERS[name](data_dir)
    return tasks


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    for name, df in load_tasks(data_dir, list(LOADERS)).items():
        print(f"{name}: {len(df)} rows, {df['group'].nunique()} groups")
        print("  conditions:", df["condition"].value_counts().to_dict())
        r = df.iloc[0]
        print("  example prompt tail:", repr(r["prompt"][-80:]))
        print("  correct:", repr(r["correct"]), "| incorrect:", repr(r["incorrect"]))
