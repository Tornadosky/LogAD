#!/usr/bin/env python3
"""
preprocess_bgl.py: Template‐mining and sequence extraction for the BGL dataset,
optimized for DeepLog, using line‐order as pseudo‐timestamp.

1. Uses Drain3 to mine log templates from each raw line.
2. Records each line’s index (“line_no”) and its cluster (EventId).
3. Saves:
   - BGL.log_templates.csv   # EventId → template text
   - log_structured.csv       # line_no, EventId per row
   - sequence.npy             # 1D array of integer‐encoded EventIds
   - vocab.json               # mapping EventId → int index (0 reserved for PAD)

This avoids failures due to unknown timestamp formats by relying on file order.
"""
import os
import json
import argparse
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

def init_drain(config_path: str) -> TemplateMiner:
    persistence = FilePersistence("drain3_bgl_state.bin")
    config = TemplateMinerConfig()
    config.load(config_path)
    config.profiling_enabled = False
    return TemplateMiner(persistence, config)

def parse_and_mine(raw_log_path: str, miner: TemplateMiner):
    """
    Reads raw BGL.log line by line, mines templates via Drain3,
    and returns:
      - df: DataFrame with ['line_no','cluster_id']
      - templates: dict cluster_id -> template string
    """
    records = []
    templates = {}
    # Count total lines for tqdm
    with open(raw_log_path, 'r', encoding='utf-8', errors='ignore') as f:
        total = sum(1 for _ in f)
    with open(raw_log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(tqdm(f, total=total, desc="Mining BGL")):
            line = line.rstrip()
            if not line:
                continue
            result = miner.add_log_message(line)
            cid = result["cluster_id"]
            tmpl = result.get("template_mined") or result.get("log_template")
            if cid not in templates:
                templates[cid] = tmpl
            records.append((line_no, cid))
    df = pd.DataFrame(records, columns=["line_no", "cluster_id"])
    return df, templates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-log",     default="data/BGL/BGL.log")
    parser.add_argument("--drain-config",default="drain3.ini")
    parser.add_argument("--out-dir",     default="data/BGL/preprocessed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    miner = init_drain(args.drain_config)

    # Mine templates & build structured table
    df_struct, templates = parse_and_mine(args.raw_log, miner)
    if df_struct.empty:
        raise RuntimeError("No lines parsed—check raw-log path.")

    # 1) Save template mapping
    templ_df = pd.DataFrame({
        "EventId":       [f"E{cid}" for cid in sorted(templates)],
        "EventTemplate": [templates[cid] for cid in sorted(templates)]
    })
    templ_df.to_csv(
        os.path.join(args.out_dir, "BGL.log_templates.csv"),
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\"
    )

    # 2) Save structured log with line_no
    df_struct["EventId"] = df_struct["cluster_id"].apply(lambda c: f"E{c}")
    df_struct[["line_no", "EventId"]].to_csv(
        os.path.join(args.out_dir, "log_structured.csv"),
        index=False
    )

    # 3) Build sequence (in file order)
    seq = df_struct.sort_values("line_no")["EventId"].tolist()

    # 4) Build vocab & encode
    uniq = sorted(set(seq))
    vocab = {e: i+1 for i, e in enumerate(uniq)}  # 1..N
    vocab["<PAD>"] = 0
    seq_int = np.array([vocab[e] for e in seq], dtype=np.int32)
    np.save(os.path.join(args.out_dir, "sequence.npy"), seq_int)

    # 5) Save vocab
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    print("✅ BGL preprocessing complete.")
    print(f"  Templates:       {len(templates)}")
    print(f"  Structured rows: {len(df_struct)}")
    print(f"  Sequence length: {len(seq_int)}")

if __name__ == "__main__":
    main()
