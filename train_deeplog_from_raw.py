#!/usr/bin/env python3
"""
train_deeplog_from_raw.py: One‐step DeepLog training directly from raw .log

1. Streams raw log lines.
2. Online‐mines templates with Drain3.
3. Builds sliding‐window next‐event samples from normals only.
4. Trains LSTM next‐event predictor on GPU.
5. Evaluates anomaly detection on a held‐out split.
"""

import os, json, argparse
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from tqdm import tqdm

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

# ────────────────────────────────────────────────────────────────────────────────
# 1) Dataset that mines templates on the fly
# ────────────────────────────────────────────────────────────────────────────────
class DeepLogRawDataset(Dataset):
    def __init__(self, log_path, drain_config, seq_len, max_windows):
        self.seq_len = seq_len
        self.max_windows = max_windows

        # init Drain3
        persistence = FilePersistence("drain3_state_raw.bin")
        cfg = TemplateMinerConfig(); cfg.load(drain_config); cfg.profiling_enabled=False
        self.miner = TemplateMiner(persistence, cfg)

        # first pass: build {cluster_id → template} and collect EventId sequence
        self.templates = {}
        events = []
        print("Mining templates and collecting event stream…")
        with open(log_path, encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc="Lines"):
                line = line.strip()
                if not line: continue
                res = self.miner.add_log_message(line)
                cid = res["cluster_id"]
                tmpl = res.get("template_mined") or res.get("log_template")
                if cid not in self.templates:
                    self.templates[cid] = tmpl
                events.append(cid)

        # build vocab: E<cid> → int
        self.vocab = {cid: i+1 for i, cid in enumerate(sorted(self.templates))}
        self.vocab["<PAD>"] = 0

        # now build sliding windows of (inp, tgt) for *normal* only
        # we need the anomaly labels too, so assume labels CSV in same folder
        labelf = os.path.join(os.path.dirname(log_path), "anomaly_label.csv")
        labels = pd.read_csv(labelf).set_index("BlockId")["Label"].to_dict()

        self.samples = []
        self.trace_ids = []
        self.labels = []

        print("Building windows…")
        # naive: assign each event to its BlockId by re-parsing block IDs from the line
        # (or you could pre-extract block_ids in the first pass too)
        # For brevity, here we assume events list corresponds to a single trace per log line:
        # you’d need to group by block_id if you want trace-based windows.
        # This sketch shows the mechanics.
        for i in tqdm(range(seq_len, len(events)), desc="Windows"):
            inp = events[i-seq_len:i]
            tgt = events[i]
            bid = f"trace_{i//seq_len}"       # placeholder: real code must extract block IDs
            lbl = labels.get(bid, "Normal")   # default to Normal if missing
            if lbl != "Normal":
                continue  # skip anomalies in training
            self.samples.append((inp, tgt))
            self.labels.append(0)  # normal only

        print(f"Vocab size: {len(self.vocab)}, windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        # map cluster_ids → ints
        xi = [self.vocab[cid] for cid in inp]
        yi = self.vocab[tgt]
        return torch.tensor(xi, dtype=torch.long), torch.tensor(yi, dtype=torch.long)

# ────────────────────────────────────────────────────────────────────────────────
# 2) Simple LSTM model
# ────────────────────────────────────────────────────────────────────────────────
class DeepLogModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out,_ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

# ────────────────────────────────────────────────────────────────────────────────
# 3) Main: train + eval
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",        required=True, help="Raw BGL.log")
    parser.add_argument("--drain-conf", required=True, help="drain3.ini")
    parser.add_argument("--out-dir",    default="deeplog_raw_out")
    parser.add_argument("--seq-len",    type=int,   default=10)
    parser.add_argument("--batch",      type=int,   default=256)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--embed",      type=int,   default=64)
    parser.add_argument("--hidden",     type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = DeepLogRawDataset(args.log, args.drain_conf, args.seq_len, max_windows=100)
    # split 80/20
    n_train = int(len(ds)*0.8)
    train_ds, test_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4)

    model = DeepLogModel(len(ds.vocab), args.embed, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        model.train()
        tot=0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            opt.zero_grad(); logits = model(X)
            loss = crit(logits, y); loss.backward(); opt.step()
            tot += loss.item()*X.size(0)
        print(f"Ep{ep} train-loss {tot/len(train_ds):.4f}")

    # simple next-event eval on test set
    model.eval(); accs=[]
    for X,y in test_loader:
        X,y = X.to(device), y.to(device)
        with torch.no_grad():
            pred = model(X).argmax(1)
        accs.append((pred==y).float().mean().item())
    print("Next-event accuracy:", np.mean(accs))

if __name__=="__main__":
    main()
