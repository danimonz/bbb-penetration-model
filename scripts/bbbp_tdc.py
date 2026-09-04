"""
Drug-discovery starter loop, v5 -- official TDC ADMET leaderboard entry.

Every earlier version evaluated on *our own* scaffold split of the MoleculeNet
BBBP CSV. That's honest, but it isn't comparable to other people's leaderboard
numbers unless everyone uses the exact same train/valid/test molecules. TDC's
BenchmarkGroup solves that: it hands out its own fixed scaffold split (the
BBB_Martins dataset -- same underlying data as MoleculeNet's BBBP, TDC's name
for it) and requires five independent training runs (seeds 1-5) so the
leaderboard reports a mean +/- std AUROC, not a single lucky number.

This version also fixes an open question from bbbp_v4.py: that script trained
a fixed 3 epochs with no way to check whether 3 was actually the best choice.
Here, each seed's run holds out TDC's validation split, checks AUROC after
every epoch, and keeps whichever epoch's weights scored highest on
validation -- instead of assuming more (or fewer) epochs is better.

Run it (needs the Python 3.11 env -- PyTDC does not install under 3.12 on
this machine, see CLAUDE.md's Environment section):
    .venv-tdc/bin/python scripts/bbbp_tdc.py
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MAX_EPOCHS = 5           # upper bound; the best-scoring epoch on validation wins, not necessarily the last one
BATCH_SIZE = 16
LR = 2e-5                # same small fine-tuning LR as v4, and for the same reason: nudge pretrained weights, don't wreck them
SEEDS = [1, 2, 3, 4, 5]  # TDC's required minimum for an official leaderboard entry

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SmilesDataset(Dataset):
    def __init__(self, smiles, labels):
        enc = tokenizer(
            list(smiles), padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }


def predict_probs(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs.extend(F.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.array(probs)


def train_one_seed(seed, train_df, valid_df, test_df, device):
    torch.manual_seed(seed)

    train_ds = SmilesDataset(train_df["Drug"].tolist(), train_df["Y"].tolist())
    valid_ds = SmilesDataset(valid_df["Drug"].tolist(), valid_df["Y"].tolist())
    test_ds = SmilesDataset(test_df["Drug"].tolist(), test_df["Y"].tolist())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    best_val_auroc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        val_probs = predict_probs(model, valid_ds, device)
        val_auroc = roc_auc_score(valid_df["Y"].tolist(), val_probs)
        print(
            f"  seed {seed} epoch {epoch + 1}/{MAX_EPOCHS}  "
            f"train loss: {total_loss / len(train_loader):.4f}  val AUROC: {val_auroc:.4f}"
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    print(f"  seed {seed}: best epoch was {best_epoch} (val AUROC {best_val_auroc:.4f})")
    model.load_state_dict(best_state)
    return predict_probs(model, test_ds, device)


def main():
    device = torch.device("cpu")
    group = admet_group(path="data/tdc")
    benchmark = group.get("BBB_Martins")
    name = benchmark["name"]
    test_df = benchmark["test"]

    predictions_list = []
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        train_df, valid_df = group.get_train_valid_split(seed=seed, benchmark="BBB_Martins")
        test_probs = train_one_seed(seed, train_df, valid_df, test_df, device)
        predictions_list.append({name: test_probs})

    results = group.evaluate_many(predictions_list)
    print("\n=== TDC BBB_Martins leaderboard result (5 seeds) ===")
    print(results)


if __name__ == "__main__":
    main()
