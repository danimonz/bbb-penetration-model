"""
Drug-discovery starter loop, v3 -- hyperparameter tuning.

Same task and scaffold split as bbbp_v2.py (predict blood-brain-barrier
penetration from SMILES), with one upgrade: instead of hand-picked
hyperparameters, each model is tuned with RandomizedSearchCV.

The one subtlety: tuning needs its own train/validation split *inside*
the training set, and if that inner split were a plain random K-fold, the
same scaffold leaked into two different molecules could land in both the
tuning-fold-train and tuning-fold-validation, inflating the CV score and
picking hyperparameters that look good for the wrong reason -- the exact
problem that made bbbp_starter.py's random split misleading. So the inner
CV is grouped by scaffold too (GroupKFold), not just the outer test split.

Run it:
    pip install pandas numpy rdkit scikit-learn xgboost
    python bbbp_v3.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# 1. LOAD -- identical to bbbp_v2.py.
# ----------------------------------------------------------------------
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
df = pd.read_csv(URL)
print("Columns:", list(df.columns))          # expect: num, name, p_np, smiles
df = df[["smiles", "p_np"]].dropna()

# ----------------------------------------------------------------------
# 2. FEATURIZE -- identical to bbbp_v2.py: 2048-bit Morgan fingerprint,
#    plus each molecule's Bemis-Murcko scaffold for grouping.
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
    return np.array(X), np.array(y), np.array(scaffolds)

X, y, scaffolds = featurize(df["smiles"], df["p_np"])
print(f"Usable molecules after parsing: {len(X)}")

# ----------------------------------------------------------------------
# 3. SCAFFOLD SPLIT -- identical to bbbp_v2.py. This outer test set is
#    touched exactly once, at the very end, after tuning is finished.
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
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
scaffolds_train = scaffolds[train_idx]        # groups for the inner CV below
print(
    f"Scaffold groups: {len(set(scaffolds))}  |  "
    f"train: {len(train_idx)}  test: {len(test_idx)}"
)

# ----------------------------------------------------------------------
# 4. TUNE -- RandomizedSearchCV with GroupKFold(5) inside the training set,
#    grouped by scaffold so no scaffold spans a tuning fold's train and
#    validation side. n_jobs=-1 lives on the search, not the base
#    estimator, so we don't nest two levels of parallelism.
# ----------------------------------------------------------------------
inner_cv = GroupKFold(n_splits=5)

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions={
        "n_estimators": randint(200, 900),
        "max_depth": [None, 10, 20, 30, 40],
        "min_samples_leaf": randint(1, 9),
        "max_features": ["sqrt", "log2", 0.3],
    },
    n_iter=25,
    scoring="roc_auc",
    cv=inner_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", random_state=42),
    param_distributions={
        "n_estimators": randint(200, 900),
        "max_depth": randint(3, 9),
        "learning_rate": uniform(0.01, 0.19),      # 0.01 .. 0.20
        "subsample": uniform(0.6, 0.4),             # 0.6 .. 1.0
        "colsample_bytree": uniform(0.5, 0.5),      # 0.5 .. 1.0
        "min_child_weight": randint(1, 6),
        "reg_lambda": uniform(0.5, 4.5),             # 0.5 .. 5.0
    },
    n_iter=25,
    scoring="roc_auc",
    cv=inner_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

searches = {"RandomForest": rf_search, "XGBoost": xgb_search}

print(f"\nTuning on {len(X_train)} molecules (5-fold grouped CV, 25 candidates each)...")
for name, search in searches.items():
    search.fit(X_train, y_train, groups=scaffolds_train)
    print(f"\n{name} best CV AUROC: {search.best_score_:.3f}")
    print(f"{name} best params: {search.best_params_}")

# ----------------------------------------------------------------------
# 5. EVALUATE -- each tuned model, once, on the held-out scaffold-split
#    test set. Compare against v2's fixed-hyperparameter numbers
#    (RandomForest 0.848, XGBoost 0.813).
# ----------------------------------------------------------------------
print(f"\nTested on {len(X_test)} molecules (scaffold split, untouched until now).")
for name, search in searches.items():
    probs = search.best_estimator_.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    print(f"{name:>12} test AUROC: {auroc:.3f}")
