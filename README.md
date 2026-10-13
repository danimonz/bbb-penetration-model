# BBB Penetration Model

A machine-learning model that predicts whether a molecule crosses the **blood-brain barrier (BBB)** from its chemical structure alone. Drug designers use this as a real filter early on, to judge whether a compound can reach the brain.

The project climbs a ladder one rung at a time: a fingerprint + RandomForest baseline, then an honest scaffold split, then hyperparameter tuning, then a fine-tuned chemical transformer, then an official entry on the [TDC ADMET leaderboard](https://tdcommons.ai/benchmark/admet_group/overview/). The last rung went back to classical ML with physicochemical descriptors, which beat the transformer.

## Headline result

**0.907 ± 0.006 test AUROC** on TDC's `BBB_Martins` benchmark. The model is a RandomForest on Morgan fingerprints plus 210 RDKit physicochemical descriptors, scored on TDC's own scaffold split and averaged over the 5 seeds TDC requires. That would place it around **9th of 27** on the [public leaderboard](https://tdcommons.ai/benchmark/admet_group/07bbb/) (snapshot from 2026-10-13; 10th place is 0.905, top entry MiniMol 0.924). It has not been submitted.

To be clear about the margin: 0.907 vs. 0.905 is well within the ± 0.006 seed spread. The accurate claim is "in the top-10 cluster," not "clearly beats 10th place."

## Results

| Version | Model | Split | Tuned | Test AUROC |
|---|---|---|---|---|
| v1 | RandomForest | random | no | 0.930 *(inflated, see below)* |
| v2 | RandomForest | scaffold (own) | no | 0.848 |
| v2 | XGBoost | scaffold (own) | no | 0.813 |
| v3 | RandomForest | scaffold (own) | yes | 0.861 |
| v3 | XGBoost | scaffold (own) | yes | 0.835 |
| v4 | ChemBERTa (fine-tuned) | scaffold (own) | — | 0.912 *(single run)* |
| v5 | ChemBERTa (fine-tuned) | scaffold (TDC official) | best epoch by validation | 0.856 ± 0.021 *(5 seeds)* |
| v6 | RandomForest, Morgan only | scaffold (TDC official) | no | 0.870 ± 0.004 *(5 seeds)* |
| v6 | RandomForest, RDKit descriptors only | scaffold (TDC official) | no | 0.903 ± 0.006 *(5 seeds)* |
| **v6** | **RandomForest, Morgan + descriptors** | **scaffold (TDC official)** | **no** | **0.907 ± 0.006** *(5 seeds)* |

**Why the numbers move the way they do:**

- **v1 → v2 (0.930 → 0.848).** A random train/test split lets near-duplicate molecules land on both sides (same ring-system core, different substituent), which inflates the score. From v2 onward the project uses a **Bemis-Murcko scaffold split**: no scaffold appears in both train and test. That's a harder and more realistic test of how well a model handles genuinely new chemistry.
- **v2 → v3.** Same scaffold split. Each model's hyperparameters are tuned with `RandomizedSearchCV`, using cross-validation grouped by scaffold, so the tuning is as honest as the final evaluation.
- **v3 → v4.** The fingerprint + classical-ML pipeline is replaced by a pretrained chemical transformer ([`seyonec/ChemBERTa-zinc-base-v1`](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)). It's fine-tuned directly on raw SMILES strings.
- **v4 → v5 (0.912 → 0.856).** Same approach, different evaluation: TDC's fixed split, an average over 5 seeds, and each seed keeps the epoch that scored best on validation. The drop is what you'd expect when a single favorable run is replaced by a seed-averaged score on a standardized split. It doesn't mean the model got worse.
- **v5 → v6 (0.856 → 0.907).** Most of the top of the leaderboard is classical models on RDKit descriptors, so I tested that directly: same RandomForest, three feature sets, one variable at a time. The set was chosen on validation AUROC (Morgan + descriptors was best on all 5 seeds) before the test scores were looked at. The explanation is chemical. Crossing the blood-brain barrier depends mostly on whole-molecule properties like polarity (TPSA), hydrogen-bond donors and lipophilicity (logP), the same properties behind medicinal chemists' [CNS MPO score](https://pmc.ncbi.nlm.nih.gov/articles/PMC3368654). Descriptors give the model those properties directly. Fingerprints and raw SMILES make it infer them from about 1,600 examples. The model's top features were TPSA, N+O count, H-bond donors, NH/OH count and logP. Notably, a plain Morgan-only RandomForest (0.870) already beats the fine-tuned transformer on TDC's split.

## What I tried that didn't work

Before v6, on TDC's split, validation AUROC averaged about 0.92 but test AUROC averaged about 0.86. I tried four changes to close that gap. To avoid trusting one lucky seed, each was screened across several seeds against the baseline, using validation AUROC only (the test set was never touched while tuning):

| Attempt | Change | Mean Δ validation AUROC vs. baseline | Verdict |
|---|---|---|---|
| 1 | Bigger backbone (`DeepChem/ChemBERTa-77M-MTR`) + LR warmup/decay | Full 5-seed test run: 0.836 ± 0.012 (worse) | Rejected |
| 2 | LR warmup/decay schedule only | +0.002 (3 seeds) | Noise, rejected |
| 3 | SMILES augmentation (2 extra random SMILES per molecule) | −0.019 (2 seeds) | Overfit faster, rejected |
| 4 | Freeze embeddings + lower 3 of 6 encoder layers | −0.003 (3 seeds) | Noise, rejected |

**Takeaways:**

1. A single seed is not enough evidence. Attempt 1 looked like a clear win on seed 1, then lost on the full 5-seed run.
2. Tuning didn't close the gap, whether the changes added capacity or removed it. The top leaderboard entries use architecturally different models, not SMILES-sequence transformers, so the next real step was a different approach rather than more hyperparameter sweeps. That turned out to be v6: classical ML with better features.

`scripts/bbbp_tdc_tune.py` is kept as a record of attempt 4, which is the last one tried.

## How it works

The core cheminformatics loop (v1–v3):

1. **Load** the MoleculeNet BBBP dataset (molecules labeled brain-penetrant or not).
2. **Featurize** each molecule's SMILES string into a 2048-bit Morgan fingerprint (ECFP4) with RDKit.
3. **Split.** v1 splits randomly. v2 and v3 split by scaffold, so train and test share no ring-system cores.
4. **Tune** (v3 only) with `RandomizedSearchCV`, scored by 5-fold cross-validation grouped by scaffold.
5. **Train** a classifier: RandomForest in v1; RandomForest and XGBoost head-to-head in v2/v3.
6. **Evaluate** with AUROC on the held-out test set.

v6 runs the same loop on TDC's split, with step 2 extended: each molecule gets its Morgan fingerprint *and* 210 RDKit physicochemical descriptors (missing/overflowing values filled from training-set medians only).

v4 and v5 replace steps 2 and 5. Instead of a fingerprint and a classical model, the raw SMILES string is tokenized with ChemBERTa's own tokenizer. Then the pretrained transformer (already trained on about 2M unlabeled SMILES) is fine-tuned on the BBB labels. v5 uses [PyTDC](https://github.com/mims-harvard/TDC)'s `admet_group` benchmark harness for the split and the scoring.

## Repository layout

```
scripts/
├── bbbp_starter.py        # v1: Morgan fingerprint + RandomForest, random split
├── bbbp_v2.py             # v2: RandomForest vs. XGBoost, scaffold split
├── bbbp_v3.py             # v3: v2 + scaffold-grouped hyperparameter tuning
├── bbbp_v4.py             # v4: fine-tuned ChemBERTa on own scaffold split
├── bbbp_tdc.py            # v5: ChemBERTa on the TDC BBB_Martins harness (5 seeds)
├── bbbp_tdc_tune.py       # ChemBERTa tuning experiment (attempt 4: frozen layers), validation only
└── bbbp_tdc_classical.py  # v6: RandomForest x 3 feature sets on TDC (current best, 0.907)
```

## Run it

**v1–v4** (Python 3.12):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/bbbp_starter.py   # v1
python scripts/bbbp_v2.py        # v2
python scripts/bbbp_v3.py        # v3
python scripts/bbbp_v4.py        # v4
```

**v5 / v6 / TDC** (Python 3.11). PyTDC won't install under Python 3.12, because its pinned scikit-learn has no 3.12 wheel. Use a separate environment:

```bash
python3.11 -m venv .venv-tdc
.venv-tdc/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv-tdc/bin/pip install pytdc==1.1.15 transformers==4.50.3

.venv-tdc/bin/python scripts/bbbp_tdc_classical.py   # v6: current best, under a minute on CPU
.venv-tdc/bin/python scripts/bbbp_tdc.py             # v5: ChemBERTa, 5 seeds
.venv-tdc/bin/python scripts/bbbp_tdc_tune.py        # ChemBERTa tuning experiment
```

Everything runs on CPU. Every script is seeded.

## Stack

Python · RDKit · scikit-learn · XGBoost · PyTorch · HuggingFace Transformers · PyTDC · pandas · NumPy

## Data

- [MoleculeNet BBBP](https://arxiv.org/abs/1703.00564) (Wu et al., 2018): blood-brain-barrier penetration dataset (v1–v4). The scripts download it from DeepChem's public copy.
- [TDC `BBB_Martins`](https://tdcommons.ai/single_pred_tasks/adme/#bbb-blood-brain-barrier-martins-et-al): the same underlying data, with TDC's fixed benchmark split (v5, v6).

## License

MIT — see [LICENSE](LICENSE).
