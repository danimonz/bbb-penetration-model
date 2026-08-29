"""
Drug-discovery starter loop, v2 -- scaffold split + model comparison.

Same task as bbbp_starter.py (predict blood-brain-barrier penetration from
SMILES), with two upgrades:
    1. SPLIT     molecules by Bemis-Murcko scaffold instead of randomly, so
                 train and test contain genuinely different chemical
                 scaffolds -- a harder, more realistic test than a random
                 split, which leaks near-duplicate structures across the
                 split and inflates AUROC.
    2. COMPARE   two models -- RandomForest and XGBoost -- on the identical
                 features and split, so any AUROC difference reflects the
                 model, not the data.

Run it:
    pip install pandas numpy rdkit scikit-learn xgboost
    python bbbp_v2.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# 1. LOAD -- identical to bbbp_starter.py.
# ----------------------------------------------------------------------
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
df = pd.read_csv(URL)
print("Columns:", list(df.columns))          # expect: num, name, p_np, smiles
df = df[["smiles", "p_np"]].dropna()

# ----------------------------------------------------------------------
# 2. FEATURIZE -- same 2048-bit Morgan fingerprint as before. This pass also
#    records each molecule's Bemis-Murcko scaffold (the ring-system "core"
#    of the molecule with side chains stripped off), since the split in
#    step 3 needs to group molecules by scaffold.
# ----------------------------------------------------------------------
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurize(smiles_series, labels):
    X, y, scaffolds = [], [], []
    for smi, label in zip(smiles_series, labels):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:                       # skip SMILES RDKit can't parse
            continue
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        X.append(arr)
        y.append(int(label))
        scaffolds.append(
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        )
    return np.array(X), np.array(y), scaffolds

X, y, scaffolds = featurize(df["smiles"], df["p_np"])
print(f"Usable molecules after parsing: {len(X)}")

# ----------------------------------------------------------------------
# 3. SCAFFOLD SPLIT -- group molecule indices by scaffold, then hand out
#    whole scaffold groups to train/test (largest groups first) until train
#    hits ~80%. No scaffold appears on both sides, so the model is tested on
#    chemistry it has never structurally seen -- this is the split used for
#    BBBP in the MoleculeNet benchmark, and it's deterministic rather than
#    random (same result every run, no random_state needed).
# ----------------------------------------------------------------------
def scaffold_split(scaffolds, frac_train=0.8):
    groups = {}
    for idx, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(idx)

    # Largest scaffold groups first (ties broken by first-seen index) so a
    # handful of common scaffolds don't get arbitrarily split across the
    # train/test boundary.
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
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(
    f"Scaffold groups: {len(set(scaffolds))}  |  "
    f"train: {len(train_idx)}  test: {len(test_idx)}"
)

# ----------------------------------------------------------------------
# 4. TRAIN + EVALUATE -- fit both models on the exact same scaffold-split
#    train set and features, then compare test AUROC on the exact same
#    test set. Any AUROC gap between them is attributable to the model.
# ----------------------------------------------------------------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

print(f"\nTrained on {len(X_train)} molecules, tested on {len(X_test)} (scaffold split).")
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    print(f"{name:>12} test AUROC: {auroc:.3f}")
