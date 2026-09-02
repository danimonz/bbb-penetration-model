"""
Drug-discovery starter loop, v4 -- fine-tune a pretrained chemical transformer.

Every earlier version (starter, v2, v3) turned each molecule into a Morgan
fingerprint -- a fixed 2048-bit vector we hand-engineered from its substructures
-- and fed that into a classical model (RandomForest / XGBoost). This version
replaces both pieces: instead of a hand-engineered feature vector, we feed the
raw SMILES string into ChemBERTa, a transformer that was already *pretrained*
on ~2 million unlabeled SMILES strings (masked-language-modeling: it learned to
guess a blanked-out chunk of a SMILES string, which forces it to internalize
chemical syntax and structure without ever seeing a BBB label). We then
*fine-tune* that pretrained model -- keep its learned weights, add a small
classification head on top, and keep training, this time on our small labeled
BBBP set. Fine-tuning a pretrained model is the standard move when your own
labeled dataset (here, ~2000 molecules) is far too small to train a transformer
from scratch.

Same data source and same scaffold split as v2/v3, so the test AUROC is
directly comparable to the tuned RandomForest's 0.861.

Run it:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers rdkit pandas numpy scikit-learn
    python bbbp_v4.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.manual_seed(42)
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

# ----------------------------------------------------------------------
# 1. LOAD -- identical source to bbbp_v2.py / bbbp_v3.py.
# ----------------------------------------------------------------------
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
df = pd.read_csv(URL)
df = df[["smiles", "p_np"]].dropna()

# ----------------------------------------------------------------------
# 2. FILTER + SCAFFOLD -- unlike v2/v3, we don't fingerprint here. The
#    transformer's own tokenizer will turn the raw SMILES string into
#    tokens later. We still need RDKit for two things: dropping SMILES
#    it can't parse (so v4 sees the same molecule set as v2/v3), and
#    computing each molecule's Bemis-Murcko scaffold for the split.
# ----------------------------------------------------------------------
smiles_list, y, scaffolds = [], [], []
for smi, label in zip(df["smiles"], df["p_np"]):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        continue
    smiles_list.append(str(smi))
    y.append(int(label))
    scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))

smiles_arr = np.array(smiles_list)
y = np.array(y)
scaffolds = np.array(scaffolds)
print(f"Usable molecules after parsing: {len(smiles_arr)}")

# ----------------------------------------------------------------------
# 3. SCAFFOLD SPLIT -- identical function to bbbp_v3.py. Deterministic
#    (no randomness), so with the same source data this produces the
#    same train/test molecules as v2/v3 did.
# ----------------------------------------------------------------------
def scaffold_split(scaffolds, frac_train=0.8):
    groups = {}
    for idx, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(idx)

    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)

    train_cutoff = frac_train * len(scaffolds)
    train_idx, test_idx = [], []
    for group in ordered:
        if len(train_idx) + len(group) > train_cutoff:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return np.array(train_idx), np.array(test_idx)

train_idx, test_idx = scaffold_split(scaffolds, frac_train=0.8)
smiles_train, smiles_test = smiles_arr[train_idx], smiles_arr[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Scaffold groups: {len(set(scaffolds))}  |  train: {len(train_idx)}  test: {len(test_idx)}")

# ----------------------------------------------------------------------
# 4. TOKENIZE -- ChemBERTa ships its own tokenizer, trained on SMILES
#    rather than English text, so it already knows chemistry-specific
#    tokens ("Cl", "=O", ring-closure digits, etc.) as single units where
#    it can. We tokenize the whole split up front since ~2000 short
#    strings easily fit in memory -- no need to tokenize on the fly.
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SmilesDataset(Dataset):
    def __init__(self, smiles, labels):
        enc = tokenizer(
            list(smiles),
            padding=True,
            truncation=True,
            max_length=128,           # generous: our SMILES are all well under this
            return_tensors="pt",
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

train_ds = SmilesDataset(smiles_train, y_train)
test_ds = SmilesDataset(smiles_test, y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# ----------------------------------------------------------------------
# 5. MODEL -- load the pretrained ChemBERTa weights, with a fresh
#    (randomly initialized) 2-class classification head on top. Every
#    weight below the head starts from the pretrained checkpoint; the
#    head starts from scratch and is what fine-tuning mostly has to
#    teach.
# ----------------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
device = torch.device("cpu")
model.to(device)

# ----------------------------------------------------------------------
# 6. FINE-TUNE -- a plain PyTorch loop (no Trainer wrapper) so every step
#    is visible. AutoModelForSequenceClassification computes cross-entropy
#    loss internally when we pass `labels`, so `outputs.loss` is ready to
#    call .backward() on directly.
# ----------------------------------------------------------------------
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)   # small LR: we're nudging pretrained weights, not learning from scratch

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        total_loss += outputs.loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS}  mean training loss: {total_loss / len(train_loader):.4f}")

# ----------------------------------------------------------------------
# 7. EVALUATE -- same held-out scaffold-split test set as v2/v3.
# ----------------------------------------------------------------------
model.eval()
test_loader = DataLoader(test_ds, batch_size=32)
all_probs, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = F.softmax(logits, dim=-1)[:, 1]   # P(class 1) = P(BBB-penetrant)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

auroc = roc_auc_score(all_labels, all_probs)
print(f"\nTested on {len(test_ds)} molecules (scaffold split, same molecules as v2/v3).")
print(f"ChemBERTa (fine-tuned) test AUROC: {auroc:.3f}")
print("Tuned RandomForest (v3) for comparison: 0.861")
