"""
Multi-seed validation-only tuning check for the TDC leaderboard fine-tune (bbbp_tdc.py).

History (2026-09-04), full writeups in drug-discovery-wiki/decisions/2026-09-04-tdc-tuning-attempts.md:
- Attempt 1: swap backbone to DeepChem/ChemBERTa-77M-MTR + linear warmup/decay
  schedule, checked against seed 1 only. Looked like a win (0.924 vs. 0.906
  val AUROC). Wasn't: the full 5-seed run scored 0.836 +/- 0.012 test AUROC,
  worse than the 0.856 +/- 0.021 baseline -- seed 1 wasn't representative.
- Attempt 2: schedule alone (proven backbone), screened across 3 seeds
  (1, 2, 4). Mean delta vs. baseline: +0.0021 -- noise-level. Not adopted.
- Attempt 3: SMILES augmentation (3 rows per training molecule: original +
  2 RDKit-random-traversal variants, same label). Screened seeds 1, 2. Mean
  delta: -0.0185 -- worse on both seeds. Diagnosis: training loss kept
  falling but validation AUROC peaked at epoch 1 and degraded from there --
  3x more rows per "epoch" means 3x the gradient updates, so it raced past
  the training budget that worked for baseline and overfit on a still-finite,
  still-correlated set of labels.

Pattern across all three: every change added *more* effective capacity or
*more* effective gradient updates relative to this tiny (~1600-molecule)
dataset, and the model overfit faster, not better. Attempt 4 (this version)
tries the opposite: REDUCE trainable capacity instead of adding it. ChemBERTa-
zinc-base-v1 is a 6-layer RoBERTa (confirmed via AutoConfig: num_hidden_layers
= 6, ~44M params total). Freezing the embeddings + the first 3 of 6 encoder
layers, and fine-tuning only the top 3 layers + the classification head, is
the standard transfer-learning move for a downstream dataset this small --
the lower layers already encode general SMILES syntax/chemistry from
pretraining; only the upper, more task-specific layers need to adapt (see
concepts/fine-tuning.md; general technique e.g. Howard & Ruder 2018 ULMFiT
gradual unfreezing, Peters et al. 2019 on partial BERT fine-tuning).

Proven backbone, flat LR (no schedule), no augmentation -- freezing is the
only new variable, checked across 3 seeds (1, 2, 4) per the now-established
methodology (a single seed isn't enough evidence, see attempt 1). Never
touches the TDC test set.

Run it (same env as bbbp_tdc.py):
    .venv-tdc/bin/python scripts/bbbp_tdc_tune.py
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bbbp_tdc import MODEL_NAME, predict_probs

MAX_EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5
FROZEN_LAYERS = 3  # freeze embeddings + encoder.layer[0:FROZEN_LAYERS]; model has 6 layers total
SEEDS_TO_CHECK = [1, 2, 4]
BASELINE_VAL_AUROC = {1: 0.9060, 2: 0.9320, 4: 0.9090}  # from 2026-09-03 decision log, same seeds

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SmilesDataset(Dataset):
    def __init__(self, smiles, labels):
        enc = tokenizer(list(smiles), padding=True, truncation=True, max_length=128, return_tensors="pt")
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


def freeze_lower_layers(model):
    """Freeze the embedding table and the first FROZEN_LAYERS encoder blocks so
    only the top layers + classification head receive gradient updates -- cuts
    trainable capacity roughly in half for this 6-layer model."""
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for layer in model.roberta.encoder.layer[:FROZEN_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")


def run_seed(seed, train_df, valid_df, device):
    torch.manual_seed(seed)
    train_ds = SmilesDataset(train_df["Drug"].tolist(), train_df["Y"].tolist())
    valid_ds = SmilesDataset(valid_df["Drug"].tolist(), valid_df["Y"].tolist())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    freeze_lower_layers(model)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    best_val_auroc = -1.0
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

    return best_epoch, best_val_auroc


def main():
    device = torch.device("cpu")
    group = admet_group(path="data/tdc")

    results = {}
    for seed in SEEDS_TO_CHECK:
        print(f"\n=== Seed {seed} (frozen-lower-layers candidate) ===")
        train_df, valid_df = group.get_train_valid_split(seed=seed, benchmark="BBB_Martins")
        best_epoch, best_val_auroc = run_seed(seed, train_df, valid_df, device)
        results[seed] = best_val_auroc
        print(f"  seed {seed}: best epoch {best_epoch}, val AUROC {best_val_auroc:.4f}")

    print("\n=== Frozen-lower-layers candidate vs. baseline (same seeds) ===")
    print(f"{'seed':>6} {'baseline':>10} {'candidate':>10} {'delta':>8}")
    for seed in SEEDS_TO_CHECK:
        base = BASELINE_VAL_AUROC[seed]
        cand = results[seed]
        print(f"{seed:>6} {base:>10.4f} {cand:>10.4f} {cand - base:>+8.4f}")

    cand_mean = np.mean(list(results.values()))
    base_mean = np.mean([BASELINE_VAL_AUROC[s] for s in SEEDS_TO_CHECK])
    print(f"{'mean':>6} {base_mean:>10.4f} {cand_mean:>10.4f} {cand_mean - base_mean:>+8.4f}")


if __name__ == "__main__":
    main()
