#!/usr/bin/env python3
"""
train_deeplog.py: Train & evaluate a DeepLog LSTM model on HDFS event sequences 
using PyTorch, with anomaly‐detection evaluation.

1. Builds sliding‐window next‐event training data from normal traces.
2. Trains an LSTM to predict the next EventId.
3. Computes per‐trace anomaly scores on a held‐out set (both normal & anomaly):
   score = max over windows of (1 − P(true_next_event)).
4. Finds a threshold on the validation normal set, then evaluates Precision/Recall/F1
   on the test set.
5. Saves model, vocab, and prints classification report + ROC/AUC.

Now with progress bars via tqdm and faster iteration via itertuples().
"""
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DeepLogDataset(Dataset):
    def __init__(self, traces_df, labels_df, seq_len, max_windows, include_anomalies=False):
        # build vocab
        all_events = set()
        for feats in traces_df['Features']:
            s = feats.strip()[1:-1]
            for ev in s.split(','):
                ev = ev.strip()
                if ev:
                    all_events.add(ev)
        self.vocab = {e: i+1 for i, e in enumerate(sorted(all_events))}
        self.vocab['<PAD>'] = 0

        self.seq_len = seq_len
        self.samples = []
        self.trace_ids = []
        self.labels = []

        normal_ids = set(labels_df[labels_df['Label']=='Normal']['BlockId'])
        anomaly_ids = set(labels_df[labels_df['Label']=='Anomaly']['BlockId'])

        # build samples with a progress bar
        for row in tqdm(traces_df.itertuples(index=False),
                        total=len(traces_df),
                        desc="Building DeepLog samples"):
            bid = row.BlockId
            is_normal = bid in normal_ids
            if not include_anomalies and not is_normal:
                continue
            if include_anomalies and bid not in normal_ids.union(anomaly_ids):
                continue
            feats = row.Features.strip()
            if not (feats.startswith('[') and feats.endswith(']')):
                continue
            events = [e.strip() for e in feats[1:-1].split(',') if e.strip()]
            idxs = [self.vocab[e] for e in events if e in self.vocab]
            if len(idxs) <= seq_len:
                continue
            windows = []
            for i in range(seq_len, len(idxs)):
                inp = idxs[i-seq_len:i]
                tgt = idxs[i]
                windows.append((inp, tgt))
            if max_windows and len(windows) > max_windows:
                windows = windows[:max_windows]
            for inp, tgt in windows:
                self.samples.append((inp, tgt))
                self.trace_ids.append(bid)
                self.labels.append(0 if bid in normal_ids else 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return (
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            self.trace_ids[idx],
            self.labels[idx]
        )

class DeepLogModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

def compute_trace_scores(model, loader, device):
    model.eval()
    scores_by_trace = {}
    labels_by_trace = {}
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch_x, batch_y, bids, labs in tqdm(loader,
                                                 desc="Scoring traces",
                                                 leave=False):
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = softmax(logits)
            true_probs = probs[range(len(batch_y)), batch_y.to(device)].cpu().numpy()
            an_scores = 1.0 - true_probs
            for bid, lab, s in zip(bids, labs, an_scores):
                scores_by_trace.setdefault(bid, []).append(s)
                labels_by_trace[bid] = lab
    trace_ids = list(scores_by_trace.keys())
    y_true, y_score = [], []
    for bid in trace_ids:
        y_true.append(labels_by_trace[bid])
        y_score.append(max(scores_by_trace[bid]))
    return np.array(y_true), np.array(y_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="data/HDFS/preprocessed")
    parser.add_argument("--output-dir", default="deeplog_output")
    parser.add_argument("--seq-len",      type=int,   default=10)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--embed-dim",    type=int,   default=64)
    parser.add_argument("--hidden-dim",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--max-windows",  type=int,   default=100)
    parser.add_argument("--val-split",    type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data
    traces = pd.read_csv(os.path.join(args.preprocessed_dir, "Event_traces.csv"))
    labels = pd.read_csv(os.path.join(args.preprocessed_dir, "anomaly_label.csv"))

    # Build full dataset (includes anomalies for eval)
    full_ds = DeepLogDataset(traces, labels,
                             seq_len=args.seq_len,
                             max_windows=args.max_windows,
                             include_anomalies=True)
    vocab = full_ds.vocab
    print(f"Vocab size: {len(vocab)}")
    print(f"Total windows: {len(full_ds)}")

    # Split indices
    idxs = np.arange(len(full_ds))
    normal_idxs = [i for i in idxs if full_ds.labels[i] == 0]
    train_idx, temp_idx = train_test_split(
        normal_idxs,
        test_size=args.val_split + 0.2,
        random_state=42
    )
    val_frac = args.val_split / (args.val_split + 0.2)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_frac,
        random_state=42
    )

    # DataLoaders
    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, val_idx),
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(Subset(full_ds, test_idx),
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # Model setup
    model = DeepLogModel(len(vocab), args.embed_dim, args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with progress bars
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y, *_ in tqdm(train_loader,
                                         desc=f"Epoch {ep}/{args.epochs} train",
                                         leave=False):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        print(f"Epoch {ep}/{args.epochs} — train loss: {total_loss/len(train_loader.dataset):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, *_ in tqdm(val_loader,
                                             desc=f"Epoch {ep}/{args.epochs} val",
                                             leave=False):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                val_loss += criterion(logits, batch_y).item() * batch_x.size(0)
        print(f"         — val loss:   {val_loss/len(val_loader.dataset):.4f}")

    # Save model & vocab
    torch.save(model.state_dict(), os.path.join(args.output_dir, "deeplog_model.pth"))
    with open(os.path.join(args.output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # Evaluation
    y_val, s_val = compute_trace_scores(model, val_loader, device)
    thresh = np.percentile(s_val[y_val == 0], 100 * (1 - args.val_split))
    print(f"\nChosen anomaly threshold: {thresh:.4f}")

    y_test, s_test = compute_trace_scores(model, test_loader, device)
    y_pred = (s_test >= thresh).astype(int)

    print("\n=== DeepLog Anomaly Detection ===")
    print(classification_report(y_test, y_pred, target_names=["Normal","Anomaly"]))
    print(f"ROC AUC: {roc_auc_score(y_test, s_test):.3f}")
    prec, rec, _ = precision_recall_curve(y_test, s_test)
    print(f"PR AUC:  {auc(rec, prec):.3f}")

    # Save evaluation data
    np.savez(os.path.join(args.output_dir, "deeplog_eval.npz"),
             y_test=y_test, scores=s_test)

if __name__ == "__main__":
    main()
