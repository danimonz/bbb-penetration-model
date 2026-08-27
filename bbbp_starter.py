"""
Drug-discovery starter loop -- no PyTDC needed.
Task: predict blood-brain-barrier (BBB) penetration from a molecule's structure.

Pulls the public MoleculeNet BBBP dataset straight from a URL (same blood-brain-
barrier data as TDC's BBB_Martins), then runs the core loop:
    1. LOAD      the dataset (a CSV of molecules + labels)
    2. FEATURIZE each SMILES string into numbers with RDKit
    3. TRAIN     a machine-learning model
    4. EVALUATE  it on molecules it never saw

Run it:
    pip install pandas numpy rdkit scikit-learn
    python bbbp_starter.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
# 1. LOAD  -- MoleculeNet's BBBP dataset. pandas reads it straight from the URL.
#    Column 'p_np' is the label: 1 = crosses into the brain, 0 = does not.
#    Column 'smiles' is the molecule's text structure.
# ----------------------------------------------------------------------
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
df = pd.read_csv(URL)
print("Columns:", list(df.columns))          # expect: num, name, p_np, smiles
df = df[["smiles", "p_np"]].dropna()

# ----------------------------------------------------------------------
# 2. FEATURIZE  -- turn each SMILES into a 2048-bit Morgan fingerprint:
#    a vector of 0s/1s marking which chemical substructures are present.
#    (A few SMILES in this dataset are malformed -- we skip those. RDKit will
#     print "SMILES Parse Error" warnings for them; that's expected and fine.)
# ----------------------------------------------------------------------
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def featurize(smiles_series, labels):
    X, y = [], []
    for smi, label in zip(smiles_series, labels):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:                       # skip SMILES RDKit can't parse
            continue
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8)
        ConvertToNumpyArray(fp, arr)
        X.append(arr)
        y.append(int(label))
    return np.array(X), np.array(y)

X, y = featurize(df["smiles"], df["p_np"])
print(f"Usable molecules after parsing: {len(X)}")

# ----------------------------------------------------------------------
# 3. TRAIN  -- hold out 20% to test on, then fit a Random Forest baseline.
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 4. EVALUATE  -- AUROC on the held-out test set (1.0 = perfect, 0.5 = coin flip).
# ----------------------------------------------------------------------
probs = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, probs)

print(f"\nTrained on {len(X_train)} molecules, tested on {len(X_test)}.")
print(f"Test AUROC: {auroc:.3f}")
print("That's one full drug-discovery loop. You just did the thing.")
