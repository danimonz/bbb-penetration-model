"""
Drug-discovery loop, v6 -- classical models on TDC's official BBB_Martins harness.

Goal: reach the TDC leaderboard top 10 (>= 0.906 AUROC; 10th place on
2026-10-13 was Lantern RADR SVM at 0.905 +/- 0.007). Our fine-tuned ChemBERTa
(bbbp_tdc.py) scores 0.856 +/- 0.021, about 19th of 27.

Why go back to classical ML: almost every top-10 entry is a classical model
fed RDKit physicochemical *descriptors* (MapLight: CatBoost on fingerprints +
200 RDKit descriptors, 0.916; Lantern: RandomForest on RDKit descriptors,
0.908). Crossing the blood-brain barrier is governed mostly by a handful of
physical properties -- polarity (TPSA), lipophilicity (logP), size (MolWt),
hydrogen-bond donors -- which RDKit computes directly. A Morgan fingerprint
only records *which substructures are present*; ChemBERTa only sees the SMILES
text. Neither is handed those properties, so both have to infer them from
~1600 examples.

This script tests that hypothesis with one variable at a time. Same model
(RandomForest, fixed hyperparameters), three feature sets:
  A. morgan       -- 2048-bit ECFP4, as in v1-v3 (step 1: classical baseline on TDC)
  B. descriptors  -- 210 RDKit physicochemical descriptors only
  C. morgan+desc  -- both, concatenated (step 2)

Honest-evaluation rules (same as bbbp_tdc.py):
- TDC's own scaffold split, TDC's required 5 seeds, TDC's evaluate_many().
- Each seed trains on TDC's train split only. The validation split is used
  to *choose* between feature sets; the test set is never used to choose.
  All three test scores are reported anyway, so nothing is hidden.

Run it (Python 3.11 env, same as bbbp_tdc.py):
    .venv-tdc/bin/python scripts/bbbp_tdc_classical.py
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule warnings

SEEDS = [1, 2, 3, 4, 5]  # TDC's required minimum for an official leaderboard entry
N_TREES = 500            # fixed, untuned: we're testing features, not hyperparameters

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DESCRIPTOR_FNS = Descriptors.descList  # list of (name, function) pairs, 210 in this RDKit


def morgan(mol):
    arr = np.zeros((2048,), dtype=np.float64)
    ConvertToNumpyArray(mfpgen.GetFingerprint(mol), arr)
    return arr


def descriptors(mol):
    # Some descriptors fail on odd molecules (e.g. partial charges on metals)
    # or return huge values (Ipc can exceed 1e30). Failures become NaN here
    # and are filled in later from the training set.
    values = []
    for _, fn in DESCRIPTOR_FNS:
        try:
            values.append(float(fn(mol)))
        except Exception:
            values.append(np.nan)
    return np.array(values)


def featurize(smiles_list):
    """Return (morgan matrix, descriptor matrix) for a list of SMILES."""
    fps, descs = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(morgan(mol))
        descs.append(descriptors(mol))
    return np.array(fps), np.array(descs)


def clean_descriptors(train, *others):
    """Make descriptor matrices safe for scikit-learn.

    Infinities -> NaN, then NaN -> that column's *training-set* median (never
    a statistic computed from validation/test, which would leak). Values are
    clipped so nothing overflows when sklearn casts to float32. Trees only care
    about value *order*, not scale, so clipping extremes changes very little.
    """
    train = np.where(np.isfinite(train), train, np.nan)
    medians = np.nan_to_num(np.nanmedian(train, axis=0))
    out = []
    for m in (train, *others):
        m = np.where(np.isfinite(m), m, np.nan)
        m = np.where(np.isnan(m), medians, m)
        out.append(np.clip(m, -1e6, 1e6))
    return out


def main():
    group = admet_group(path="data/tdc")
    benchmark = group.get("BBB_Martins")
    name = benchmark["name"]
    test_df = benchmark["test"]
    X_test_fp, X_test_desc = featurize(test_df["Drug"])

    feature_sets = ["morgan", "descriptors", "morgan+desc"]
    predictions = {fs: [] for fs in feature_sets}
    val_scores = {fs: [] for fs in feature_sets}

    for seed in SEEDS:
        train_df, valid_df = group.get_train_valid_split(seed=seed, benchmark="BBB_Martins")
        X_tr_fp, X_tr_desc = featurize(train_df["Drug"])
        X_va_fp, X_va_desc = featurize(valid_df["Drug"])
        X_tr_desc, X_va_desc, X_te_desc = clean_descriptors(X_tr_desc, X_va_desc, X_test_desc)

        matrices = {
            "morgan": (X_tr_fp, X_va_fp, X_test_fp),
            "descriptors": (X_tr_desc, X_va_desc, X_te_desc),
            "morgan+desc": (
                np.hstack([X_tr_fp, X_tr_desc]),
                np.hstack([X_va_fp, X_va_desc]),
                np.hstack([X_test_fp, X_te_desc]),
            ),
        }

        print(f"\n=== Seed {seed}  (train {len(train_df)}, valid {len(valid_df)}) ===")
        for fs, (X_tr, X_va, X_te) in matrices.items():
            model = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
            model.fit(X_tr, train_df["Y"])
            val_auroc = roc_auc_score(valid_df["Y"], model.predict_proba(X_va)[:, 1])
            val_scores[fs].append(val_auroc)
            predictions[fs].append({name: model.predict_proba(X_te)[:, 1]})
            print(f"  {fs:>12}  val AUROC: {val_auroc:.4f}")

    print("\n=== Summary (5 seeds) ===")
    print(f"{'features':>12}  {'val AUROC':>16}  {'TDC test AUROC':>16}")
    for fs in feature_sets:
        test_mean, test_std = group.evaluate_many(predictions[fs])[name]
        print(
            f"{fs:>12}  {np.mean(val_scores[fs]):.3f} +/- {np.std(val_scores[fs]):.3f}"
            f"  {test_mean:.3f} +/- {test_std:.3f}"
        )
    print("\nPick by validation, not test. Top-10 cutoff: 0.906. ChemBERTa baseline: 0.856 +/- 0.021.")


if __name__ == "__main__":
    main()
