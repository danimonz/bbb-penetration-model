"""
How much of v6's TDC score is test-set luck? A bootstrap confidence interval.

v6 (bbbp_tdc_classical.py) reports 0.907 +/- 0.006 AUROC. That +/- comes from
re-running with 5 seeds. In TDC's ADMET group the seed changes the train/valid
split and the model, but the TEST SET IS FIXED: the same ~400 molecules every
time. So the +/- only measures training noise. It says nothing about how much
the score depends on which 400 molecules happened to land in the test set.

The bootstrap estimates that second source of noise:
  1. Draw 400 test molecules *with replacement* from the real 400 (some appear
     twice, some not at all). This is a plausible "other test set" from the
     same population.
  2. Score the already-trained models on that resample (TDC metric: AUROC per
     seed, then the mean over the 5 seeds).
  3. Repeat 2,000 times. The spread of those 2,000 scores is the uncertainty
     due to test-set sampling. The middle 95% is the 95% confidence interval.

No retraining happens inside the loop, only re-scoring, so this is fast.

Paired comparison: we also score descriptors-only on the SAME resamples and
look at the difference (morgan+desc minus descriptors). Using the same
resample for both models cancels out "this resample happened to be easy",
so a paired difference is much tighter than comparing two separate CIs.
If the 95% interval for the difference includes 0, the gain from adding
Morgan bits is not distinguishable from noise.

Run it (Python 3.11 env):
    .venv-tdc/bin/python scripts/bbbp_tdc_bootstrap.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group

# Reuse v6's exact featurization so we're measuring the same model.
from bbbp_tdc_classical import SEEDS, N_TREES, featurize, clean_descriptors

N_BOOT = 2000
BOOT_SEED = 0  # seeds the resampling itself, so the CI is reproducible

# Reference points from the TDC BBB_Martins leaderboard (2026-10-13).
LEADERBOARD = {"MiniMol (#1)": 0.924, "MapLight (#3)": 0.916, "10th place": 0.905}


def train_predictions(group, X_test_fp, X_test_desc):
    """Train v6's RF for each seed; return test probabilities per feature set.

    Output: {"morgan+desc": array (5 seeds, n_test), "descriptors": ...}
    """
    preds = {"morgan+desc": [], "descriptors": []}
    for seed in SEEDS:
        train_df, _ = group.get_train_valid_split(seed=seed, benchmark="BBB_Martins")
        X_tr_fp, X_tr_desc = featurize(train_df["Drug"])
        X_tr_desc, X_te_desc = clean_descriptors(X_tr_desc, X_test_desc)

        for fs, X_tr, X_te in [
            ("morgan+desc", np.hstack([X_tr_fp, X_tr_desc]), np.hstack([X_test_fp, X_te_desc])),
            ("descriptors", X_tr_desc, X_te_desc),
        ]:
            model = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
            model.fit(X_tr, train_df["Y"])
            preds[fs].append(model.predict_proba(X_te)[:, 1])
        print(f"  seed {seed} trained")
    return {fs: np.array(p) for fs, p in preds.items()}


def tdc_score(y, seed_preds, idx):
    """TDC's metric on the rows `idx`: AUROC for each seed, averaged."""
    return np.mean([roc_auc_score(y[idx], p[idx]) for p in seed_preds])


def main():
    group = admet_group(path="data/tdc")
    test_df = group.get("BBB_Martins")["test"]
    y = test_df["Y"].to_numpy()
    n = len(y)
    print(f"Test set: {n} molecules, {y.mean():.0%} labelled BBB+")

    X_test_fp, X_test_desc = featurize(test_df["Drug"])
    preds = train_predictions(group, X_test_fp, X_test_desc)

    everything = np.arange(n)
    point = {fs: tdc_score(y, p, everything) for fs, p in preds.items()}

    rng = np.random.default_rng(BOOT_SEED)
    boot = {fs: [] for fs in preds}
    while len(boot["morgan+desc"]) < N_BOOT:
        idx = rng.integers(0, n, size=n)  # n draws with replacement
        if len(np.unique(y[idx])) < 2:   # AUROC needs both classes present
            continue
        for fs, p in preds.items():
            boot[fs].append(tdc_score(y, p, idx))
    boot = {fs: np.array(v) for fs, v in boot.items()}

    print(f"\n=== Bootstrap over the test set ({N_BOOT} resamples) ===")
    for fs in preds:
        lo, hi = np.percentile(boot[fs], [2.5, 97.5])
        print(f"{fs:>12}: {point[fs]:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   "
              f"bootstrap std {boot[fs].std():.3f}")

    print("\nWhere the leaderboard sits relative to v6's interval:")
    v6 = boot["morgan+desc"]
    for label, score in LEADERBOARD.items():
        frac = (v6 >= score).mean()
        print(f"  {label:>14} {score:.3f}: v6 scores at or above it in {frac:.0%} of resamples")

    diff = boot["morgan+desc"] - boot["descriptors"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"\nPaired difference, morgan+desc minus descriptors-only: "
          f"{point['morgan+desc'] - point['descriptors']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("  -> " + ("interval excludes 0: the Morgan bits help." if lo > 0 or hi < 0
                     else "interval includes 0: the gain is within noise."))


if __name__ == "__main__":
    main()
