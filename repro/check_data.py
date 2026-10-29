#!/usr/bin/env python3
"""
Stage 1 data checks for the LOCUS reproduction.

Reads the raw EmbedLLM CSVs and the concise parquets built by
locus_code_release/data/make_concise_split.py, and writes results/data_stats.md.

Checks:
  1. concise parquets: prompts, label columns, NaN labels, embedding dims, positive rate
  2. prompts per merged benchmark per split (test compared with the sizes implied by paper Table 5)
  3. raw CSVs: duplicate (prompt_id, model_id) pairs and conflicting labels
  4. prompt overlap between splits (by prompt_id and by exact prompt text)
  5. sentence-encoder truncation (prompts longer than each encoder's max_seq_length)
  6. held-out models of Fig. 3 are model indices 96-111

Usage (from the project root, venv active):
  $env:USE_TF = "0"; python repro/check_data.py
  python repro/check_data.py --skip-truncation      # faster, no tokenizers
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "locus_code_release" / "data" / "datasets" / "embedllm"
RAW, CONCISE = DATA / "raw", DATA / "concise"
SPLITS = ["train", "val", "test"]

# Test-set sizes implied by the percentages in paper Table 5 (all ten verified on test.csv)
PAPER_TEST_COUNTS = {
    "mmlu": 1178, "gpqa": 500, "medmcqa": 354, "mathqa": 237, "asdiv": 198,
    "social_iqa": 162, "piqa": 134, "gsm8k": 112, "truthfulqa_mc1": 74, "logiqa": 51,
}
PAPER_NAMES = {
    "mmlu": "MMLU", "gpqa": "GPQA", "medmcqa": "MedQA", "mathqa": "MathQA", "asdiv": "ASDiv",
    "social_iqa": "SoQA", "piqa": "PIQA", "gsm8k": "GSM8k", "truthfulqa_mc1": "TruthQA", "logiqa": "LogiQA",
}
ENCODERS = {
    "emb_all_mpnet_base_v2": "sentence-transformers/all-mpnet-base-v2",
    "emb_paraphrase_albert_small_v2": "sentence-transformers/paraphrase-albert-small-v2",
    "emb_all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "emb_all_distilroberta_v1": "sentence-transformers/all-distilroberta-v1",
}
HELD_OUT = [
    "FelixChao/llama2-13b-math1.2", "MaziyarPanahi/WizardLM-Math-70B-v0.1", "01-ai/Yi-6B-200K",
    "WizardLM/WizardLM-70B-V1.0", "bigscience/bloom-7b1", "sail/Sailor-7B",
    "codellama/CodeLlama-13b-Instruct-hf", "Writer/palmyra-med-20b", "Qwen/Qwen1.5-0.5B-Chat",
    "databricks/dolly-v2-12b", "nomic-ai/gpt4all-13b-snoozy", "stabilityai/stablelm-tuned-alpha-7b",
    "AdaptLLM/medicine-chat", "AdaptLLM/medicine-LLM", "EleutherAI/pythia-12b", "Q-bert/Optimus-7B",
]


def merge_benchmark(category: str) -> str:
    """Same merging as scripts/evaluation.py: mmlu* -> mmlu, gpqa* -> gpqa."""
    s = str(category).strip().lower()
    for pref in ("mmlu", "gpqa"):
        if s.startswith(pref):
            return pref
    return s


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def hf_revision(fallback: str) -> str:
    ref = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--RZ412--EmbedLLM" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else f"{fallback} (recorded at download; HF cache since removed)"


def load_concise(split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    path = CONCISE / f"concise.{split}.parquet"
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    emb_cols = [c for c in names if c.startswith("emb_")]
    first = pf.read_row_group(0, columns=emb_cols).slice(0, 1).to_pydict()
    emb_dims = {c: len(first[c][0]) for c in emb_cols}
    keep = [c for c in names if not c.startswith("emb_")]
    return pd.read_parquet(path, columns=keep), emb_dims


def check_concise(frames, emb_dims) -> tuple[str, list[str]]:
    rows, problems = [], []
    for split, df in frames.items():
        lab = [c for c in df.columns if c.startswith("label_")]
        Y = df[lab].to_numpy(dtype=float)
        acc = np.nanmean(Y, axis=0)
        n_nan = int(np.isnan(Y).sum())
        rows.append({
            "split": split, "prompts": len(df), "unique prompt_id": df["prompt_id"].nunique(),
            "label cols": len(lab), "NaN labels": n_nan, "positive rate": float(np.nanmean(Y)),
            "model acc min": float(acc.min()), "model acc max": float(acc.max()),
            "emb dims": ", ".join(f"{k.removeprefix('emb_')}={v}" for k, v in emb_dims[split].items()),
        })
        if len(lab) != 112:
            problems.append(f"{split}: {len(lab)} label columns (expected 112)")
        if n_nan:
            problems.append(f"{split}: {n_nan} NaN labels (load_split casts them to 0)")
        if df["prompt_id"].nunique() != len(df):
            problems.append(f"{split}: duplicate prompt_id rows in concise parquet")
    return md_table(pd.DataFrame(rows)), problems


def check_benchmarks(frames) -> tuple[str, list[str]]:
    counts = {s: df["category"].map(merge_benchmark).value_counts() for s, df in frames.items()}
    bench = sorted(set().union(*[c.index for c in counts.values()]), key=lambda b: -counts["test"].get(b, 0))
    rows, problems = [], []
    for b in bench:
        n_test = int(counts["test"].get(b, 0))
        paper = PAPER_TEST_COUNTS.get(b)
        rows.append({
            "benchmark": b, "paper column": PAPER_NAMES.get(b, "?"),
            "train": int(counts["train"].get(b, 0)), "val": int(counts["val"].get(b, 0)), "test": n_test,
            "paper test (implied)": paper if paper is not None else "-",
            "match": "yes" if paper == n_test else "NO",
        })
        if paper != n_test:
            problems.append(f"test count for {b}: {n_test} vs paper-implied {paper}")
    n_cat = {s: df["category"].nunique() for s, df in frames.items()}
    rows.append({"benchmark": "**total**", "paper column": "", **{s: len(frames[s]) for s in SPLITS},
                 "paper test (implied)": sum(PAPER_TEST_COUNTS.values()), "match": ""})
    note = f"\nRaw categories per split: {n_cat}. `mmlu*` and `gpqa*` are merged as in `scripts/evaluation.py`.\n"
    return md_table(pd.DataFrame(rows)) + "\n" + note, problems


def check_raw_duplicates() -> str:
    rows = []
    for split in SPLITS:
        df = pd.read_csv(RAW / f"{split}.csv", usecols=["prompt_id", "model_id", "label"])
        n_missing_mid = int(df["model_id"].isna().sum())
        key = ["prompt_id", "model_id"]
        dup_mask = df.duplicated(key, keep=False)
        dup = df[dup_mask]
        n_pairs = df.drop_duplicates(key).shape[0]
        g = dup.groupby(key)["label"].nunique()
        rows.append({
            "split": split, "raw rows": len(df), "unique pairs": n_pairs,
            "extra rows": len(df) - n_pairs, "duplicated pairs": len(g),
            "pairs with conflicting labels": int((g > 1).sum()),
            "conflicting / all pairs": float((g > 1).sum() / n_pairs), "missing model_id": n_missing_mid,
        })
        del df, dup
    note = ("\n`make_concise_split.py` keeps the **last** occurrence of each (prompt_id, model_id) pair, "
            "so for conflicting pairs the label in the parquet is the last row's label.\n")
    return md_table(pd.DataFrame(rows)) + "\n" + note


def check_overlap(frames) -> str:
    rows = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ids = set(frames[a]["prompt_id"]) & set(frames[b]["prompt_id"])
        txt = set(frames[a]["prompt"]) & set(frames[b]["prompt"])
        rows.append({"splits": f"{a} ∩ {b}", "shared prompt_id": len(ids), "shared exact prompt text": len(txt)})
    return md_table(pd.DataFrame(rows))


def check_truncation(frames) -> str:
    from sentence_transformers import SentenceTransformer

    rows = []
    for col, hf_name in ENCODERS.items():
        st = SentenceTransformer(hf_name, device="cpu")
        max_len, tok = st.max_seq_length, st.tokenizer
        for split, df in frames.items():
            bench = df["category"].map(merge_benchmark).to_numpy()
            lengths = np.array([len(ids) for ids in tok(df["prompt"].astype(str).tolist(),
                                                         add_special_tokens=True, verbose=False)["input_ids"]])
            over = lengths > max_len
            per_bench = pd.Series(over).groupby(bench).mean()
            worst = ", ".join(f"{b} {v:.0%}" for b, v in per_bench.sort_values(ascending=False).items() if v > 0)
            rows.append({
                "encoder": col.removeprefix("emb_"), "max_seq_length": max_len, "split": split,
                "median tokens": int(np.median(lengths)), "truncated": float(over.mean()),
                "truncated by benchmark": worst or "none",
            })
        del st
    note = ("\nTruncated prompts are embedded from their first `max_seq_length` tokens only. GSM8K prompts are "
            "few-shot with the actual question last, so truncation removes the question itself.\n")
    return md_table(pd.DataFrame(rows)) + "\n" + note


def check_held_out() -> tuple[str, list[str]]:
    mo = pd.read_csv(CONCISE / "model_order.csv")
    name2id = dict(zip(mo["model_name"], mo["model_id"]))
    ids = [name2id.get(h.replace("/", "__")) for h in HELD_OUT]
    ok = ids == list(range(96, 112))
    df = pd.DataFrame({"held-out model (paper Fig. 3)": HELD_OUT, "model_id": [str(i) for i in ids]})
    verdict = f"\nAll 16 held-out models are indices 96-111: **{'yes' if ok else 'NO'}**. " \
              f"Total models in model_order.csv: {len(mo)}.\n"
    return md_table(df) + "\n" + verdict, ([] if ok else ["held-out models are not indices 96-111"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "data_stats.md"))
    ap.add_argument("--revision", default="266ae3e180ccb533c86afc9cfb2f5db99d171aea",
                    help="dataset revision to record if the HF cache is gone")
    ap.add_argument("--skip-truncation", action="store_true", help="skip the tokenizer-based truncation check")
    ap.add_argument("--skip-raw", action="store_true", help="skip reading the raw CSVs (duplicate check)")
    args = ap.parse_args()

    print("[1/6] loading concise parquets")
    loaded = {s: load_concise(s) for s in SPLITS}
    frames = {s: v[0] for s, v in loaded.items()}
    emb_dims = {s: v[1] for s, v in loaded.items()}

    problems: list[str] = []
    sections = []

    t, p = check_concise(frames, emb_dims); problems += p
    sections.append(("1. Concise parquets", t))
    print("[2/6] benchmark counts")
    t, p = check_benchmarks(frames); problems += p
    sections.append(("2. Prompts per benchmark", t))
    if not args.skip_raw:
        print("[3/6] raw duplicates (reads the 2.7 GB train.csv)")
        sections.append(("3. Duplicate (prompt, model) rows in the raw CSVs", check_raw_duplicates()))
    print("[4/6] split overlap")
    sections.append(("4. Prompt overlap between splits", check_overlap(frames)))
    if not args.skip_truncation:
        print("[5/6] encoder truncation (tokenizes all prompts with 4 tokenizers)")
        sections.append(("5. Sentence-encoder truncation", check_truncation(frames)))
    print("[6/6] held-out models")
    t, p = check_held_out(); problems += p
    sections.append(("6. Held-out models (Fig. 3)", t))

    header = [
        "# EmbedLLM data checks (Stage 1)", "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `repro/check_data.py`.", "",
        f"- Dataset: `RZ412/EmbedLLM`, revision `{hf_revision(args.revision)}`",
        f"- Raw CSVs: `{RAW.relative_to(PROJECT_ROOT).as_posix()}`",
        f"- Concise parquets: `{CONCISE.relative_to(PROJECT_ROOT).as_posix()}`", "",
        "## Problems found", "",
        *([f"- {p}" for p in problems] or ["- none"]), "",
    ]
    body = [f"## {title}\n\n{text}\n" for title, text in sections]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + body), encoding="utf-8")
    print(f"[done] wrote {out}  ({len(problems)} problem(s))")


if __name__ == "__main__":
    main()
