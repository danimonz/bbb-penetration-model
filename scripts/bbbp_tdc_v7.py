"""
Drug-discovery loop, v7 -- which feature groups actually carry the BBB signal?

v6 (bbbp_tdc_classical.py): RandomForest on Morgan bits + 210 RDKit descriptors,
0.907 +/- 0.006 on TDC BBB_Martins. The bootstrap (bbbp_tdc_bootstrap.py)
showed two things: the test-set 95% CI is about [0.86, 0.94], and adding Morgan
bits to the descriptors gave no gain beyond noise. So a single leaderboard
number can't tell close models apart. v7 asks a better question: *what drives
the prediction*, with every comparison carrying a paired-bootstrap interval.

Feature groups (all computed by RDKit from the SMILES string):
  desc    210 RDKit physicochemical descriptors (TPSA, logP, MolWt, ...).
  morgan  ECFP4 *count* fingerprint, 2048 slots. v6 used bits (present/absent);
          counts also record how many times each substructure occurs.
  avalon  Avalon count fingerprint, 1024 slots. A different substructure
          enumeration (paths and atom features) from Morgan's circular one.
  erg     Extended reduced graph (ErG), 315 values. Collapses the molecule
          into pharmacophore nodes (H-bond donor, acceptor, aromatic ring,
          charged group, ...) and records distances between node types.
          Describes the *shape of the interaction pattern*, not exact atoms.
  maccs   MACCS keys, 167 bits. A fixed dictionary of yes/no structural
          questions ("is there a ring with an N?").

Stages:
  1. Model + feature-set choice. RandomForest vs CatBoost, each on
     {desc, desc+morgan (v6's set), all five groups}. The final model is
     the one with the best *mean validation AUROC*. The test set is never
     used to choose.
  2. Group ablation with the chosen model: each group alone, and all groups
     with one group left out.
  3. Descriptor-family ablation: split the 210 descriptors into chemistry
     families (polarity/H-bond, lipophilicity, charge, size, topology,
     fragments) and test each alone and left out, using descriptors only.

Stages 2-3 are analysis, reported on test with a paired bootstrap against the
final model. They do not feed back into choosing the submission.

Every test-set prediction is saved to data/v7/predictions.npz so the analysis
can be re-run without retraining.

Run it (Python 3.11 env):
    .venv-tdc/bin/python scripts/bbbp_tdc_v7.py
"""

import os
import time

import numpy as np
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdReducedGraphs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group

from bbbp_tdc_classical import clean_descriptors, descriptors

RDLogger.DisableLog("rdApp.*")

SEEDS = [1, 2, 3, 4, 5]  # TDC's required 5 seeds
N_BOOT = 2000
BOOT_SEED = 0
OUT_DIR = "data/v7"
GROUPS = ["desc", "morgan", "avalon", "erg", "maccs"]

# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize_group(mol, group):
    if group == "desc":
        return descriptors(mol)  # v6's function: NaN where RDKit fails
    if group == "morgan":
        return morgan_gen.GetCountFingerprintAsNumPy(mol).astype(float)
    if group == "avalon":
        return np.array(pyAvalonTools.GetAvalonCountFP(mol, nBits=1024).ToList(), float)
    if group == "erg":
        return np.array(rdReducedGraphs.GetErGFingerprint(mol), float)
    if group == "maccs":
        return np.array(list(MACCSkeys.GenMACCSKeys(mol)), float)
    raise ValueError(group)


def featurize(smiles_list):
    """{group: matrix (n_molecules, n_features)} for a list of SMILES."""
    rows = {g: [] for g in GROUPS}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        for g in GROUPS:
            rows[g].append(featurize_group(mol, g))
    return {g: np.array(r) for g, r in rows.items()}


# ---------------------------------------------------------------------------
# Descriptor families. Every one of RDKit's 210 descriptors goes into exactly
# one family; family_columns() raises an error if one is missed or duplicated.
# ---------------------------------------------------------------------------

def descriptor_family(name):
    # Polarity / hydrogen bonding: the classic CNS rules (TPSA, HBD, N+O count).
    if name in {"TPSA", "NHOHCount", "NOCount", "NumHAcceptors", "NumHDonors", "NumHeteroatoms"}:
        return "polarity_hbond"
    # Lipophilicity: Crippen logP / molar refractivity and their surface-area splits.
    if name in {"MolLogP", "MolMR"} or name.startswith(("SlogP_VSA", "SMR_VSA")):
        return "lipophilicity"
    # Charge / electronic: partial charges and electrotopological state.
    if (name.endswith("PartialCharge") or name.endswith("EStateIndex")
            or name.startswith(("PEOE_VSA", "EState_VSA", "VSA_EState", "BCUT2D_CHG"))):
        return "charge"
    # Size and ring/flexibility counts.
    if name in {"MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
                "NumRadicalElectrons", "HeavyAtomCount", "LabuteASA",
                "NumRotatableBonds", "FractionCSP3", "RingCount"} or (
            name.startswith("Num") and name.endswith(("Rings", "cycles"))):
        return "size"
    # Fragment counts: one per functional group (fr_amide, fr_COO, ...).
    if name.startswith("fr_"):
        return "fragments"
    # Everything else: graph-topology indices (Chi, Kappa, BertzCT, BCUT2D
    # mass/logP/MR, ...) plus qed and SPS, which are composite scores.
    return "topology_other"


DESC_NAMES = [n for n, _ in Descriptors.descList]
FAMILIES = ["polarity_hbond", "lipophilicity", "charge", "size", "fragments", "topology_other"]


def family_columns():
    cols = {f: [] for f in FAMILIES}
    for i, name in enumerate(DESC_NAMES):
        cols[descriptor_family(name)].append(i)
    assert sum(len(c) for c in cols.values()) == len(DESC_NAMES)
    return {f: np.array(c) for f, c in cols.items()}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model(kind, seed):
    # Both untuned on purpose: v7 compares features, not hyperparameters.
    if kind == "rf":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    if kind == "catboost":
        # CatBoost: gradient-boosted trees (like XGBoost) with "ordered
        # boosting" to reduce overfitting on small datasets. Defaults:
        # 1000 trees, depth 6, learning rate picked from the data size.
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=-1,
                                  allow_writing_files=False)  # no catboost_info/ logs
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_config(name, kind, build, splits, y_test):
    """Train one (model, feature set) on all 5 seeds.

    `build(feats)` turns a {group: matrix} dict into the model's input matrix.
    Returns mean validation AUROC and the (5, n_test) test-probability array.
    """
    val, test = [], []
    for seed, (Xtr, ytr, Xva, yva, Xte) in zip(SEEDS, splits):
        model = make_model(kind, seed)
        model.fit(build(Xtr), ytr)
        val.append(roc_auc_score(yva, model.predict_proba(build(Xva))[:, 1]))
        test.append(model.predict_proba(build(Xte))[:, 1])
    test = np.array(test)
    test_scores = [roc_auc_score(y_test, p) for p in test]
    print(f"  {name:<34} val {np.mean(val):.3f}   test {np.mean(test_scores):.3f} "
          f"+/- {np.std(test_scores):.3f}")
    return {"val": np.mean(val), "test": test}


def stack(groups, desc_cols=None):
    """Return a `build` function that concatenates the chosen groups.

    desc_cols: if given, keep only those descriptor columns (family ablation).
    """
    def build(feats):
        parts = []
        for g in groups:
            m = feats[g]
            if g == "desc" and desc_cols is not None:
                m = m[:, desc_cols]
            parts.append(m)
        return np.hstack(parts)
    return build


def paired_bootstrap(y, preds, reference):
    """For each config: point score, 95% CI, and paired difference vs reference."""
    rng = np.random.default_rng(BOOT_SEED)
    resamples = []
    while len(resamples) < N_BOOT:
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) == 2:
            resamples.append(idx)

    def score(p, idx):
        return np.mean([roc_auc_score(y[idx], s[idx]) for s in p])

    boot = {k: np.array([score(p, idx) for idx in resamples]) for k, p in preds.items()}
    out = {}
    for k in preds:
        d = boot[k] - boot[reference]
        out[k] = {
            "point": score(preds[k], np.arange(len(y))),
            "ci": np.percentile(boot[k], [2.5, 97.5]),
            "diff": score(preds[k], np.arange(len(y))) - score(preds[reference], np.arange(len(y))),
            "diff_ci": np.percentile(d, [2.5, 97.5]),
        }
    return out


def print_table(title, keys, stats, reference):
    print(f"\n=== {title} (paired vs {reference}) ===")
    print(f"{'config':<34} {'test':>6}  {'95% CI':>15}   {'diff':>7}  {'diff 95% CI':>17}  verdict")
    for k in keys:
        s = stats[k]
        lo, hi = s["diff_ci"]
        verdict = "-" if k == reference else ("worse" if hi < 0 else "better" if lo > 0 else "tie")
        print(f"{k:<34} {s['point']:.3f}  [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]   "
              f"{s['diff']:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")


def main():
    start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    group = admet_group(path="data/tdc")
    bench = group.get("BBB_Martins")
    name = bench["name"]
    test_df, pool = bench["test"], bench["train_val"]
    y_test = test_df["Y"].to_numpy()

    # Featurize every molecule once; each seed's split is just a row lookup.
    print("Featurizing...")
    test_feats = featurize(test_df["Drug"])
    pool_feats = featurize(pool["Drug"])
    row_of = {smi: i for i, smi in enumerate(pool["Drug"])}

    splits = []
    for seed in SEEDS:
        tr, va = group.get_train_valid_split(seed=seed, benchmark="BBB_Martins")
        tri = [row_of[s] for s in tr["Drug"]]
        vai = [row_of[s] for s in va["Drug"]]
        ftr = {g: pool_feats[g][tri] for g in GROUPS}
        fva = {g: pool_feats[g][vai] for g in GROUPS}
        fte = dict(test_feats)
        # Fill descriptor NaNs from this seed's *training* rows only (no leakage).
        ftr["desc"], fva["desc"], fte["desc"] = clean_descriptors(ftr["desc"], fva["desc"], fte["desc"])
        splits.append((ftr, tr["Y"].to_numpy(), fva, va["Y"].to_numpy(), fte))

    results = {}

    # ---- Stage 1: model + feature-set choice, by validation ----------------
    print("\n--- Stage 1: model and feature set (chosen by validation) ---")
    stage1_sets = {"desc": ["desc"], "desc+morgan": ["desc", "morgan"], "all": GROUPS}
    for kind in ["rf", "catboost"]:
        for set_name, groups in stage1_sets.items():
            key = f"{kind} | {set_name}"
            results[key] = run_config(key, kind, stack(groups), splits, y_test)
    stage1 = [k for k in results]
    final = max(stage1, key=lambda k: results[k]["val"])
    final_kind = final.split(" | ")[0]
    print(f"\nChosen by validation: {final}")

    # Official TDC number for the chosen model.
    tdc_mean, tdc_std = group.evaluate_many([{name: p} for p in results[final]["test"]])[name]
    print(f"TDC evaluate_many: {tdc_mean:.3f} +/- {tdc_std:.3f}")

    # ---- Stage 2: group ablation with the chosen model ---------------------
    print(f"\n--- Stage 2: feature-group ablation ({final_kind}) ---")
    stage2 = []
    for g in GROUPS:
        for key, groups in [(f"{final_kind} | only {g}", [g]),
                            (f"{final_kind} | all minus {g}", [x for x in GROUPS if x != g])]:
            if key not in results:
                results[key] = run_config(key, final_kind, stack(groups), splits, y_test)
            stage2.append(key)
    all_key = f"{final_kind} | all"

    # ---- Stage 3: descriptor-family ablation (descriptors only) ------------
    print(f"\n--- Stage 3: descriptor families ({final_kind}, descriptors only) ---")
    fam = family_columns()
    print("  family sizes: " + ", ".join(f"{f} {len(c)}" for f, c in fam.items()))
    desc_key = f"{final_kind} | desc"
    stage3 = []
    for f in FAMILIES:
        rest = np.concatenate([fam[x] for x in FAMILIES if x != f])
        for key, cols in [(f"{final_kind} | desc: only {f}", fam[f]),
                          (f"{final_kind} | desc minus {f}", rest)]:
            results[key] = run_config(key, final_kind, stack(["desc"], cols), splits, y_test)
            stage3.append(key)

    np.savez(os.path.join(OUT_DIR, "predictions.npz"), y_test=y_test,
             **{k: v["test"] for k, v in results.items()})

    # ---- Paired bootstrap tables -------------------------------------------
    print(f"\nBootstrapping ({N_BOOT} test resamples)...")
    preds = {k: v["test"] for k, v in results.items()}
    stats_final = paired_bootstrap(y_test, preds, final)
    print_table("Stage 1", stage1, stats_final, final)

    stats_all = paired_bootstrap(y_test, preds, all_key)
    print_table("Stage 2: groups", [all_key] + stage2, stats_all, all_key)

    stats_desc = paired_bootstrap(y_test, preds, desc_key)
    print_table("Stage 3: descriptor families", [desc_key] + stage3, stats_desc, desc_key)

    print("\nValidation AUROC for every config (what a choice would be based on):")
    for k, v in results.items():
        print(f"  {k:<34} {v['val']:.3f}")
    print(f"\nDone in {(time.time() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
