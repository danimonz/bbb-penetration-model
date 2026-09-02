# BBB Penetration Model

A machine-learning model that predicts whether a molecule crosses the **blood-brain barrier (BBB)** from its chemical structure alone — a real filter used in early drug design to judge whether a compound can reach the brain.

## Results

| Version | Model | Split | Tuned | Test AUROC |
|---|---|---|---|---|
| v1 | RandomForest | random | no | 0.930 |
| v2 | RandomForest | scaffold | no | 0.848 |
| v2 | XGBoost | scaffold | no | 0.813 |
| v3 | RandomForest | scaffold | yes | 0.861 |
| v3 | XGBoost | scaffold | yes | 0.835 |
| v4 | ChemBERTa (fine-tuned) | scaffold | — | **0.912** |

The v1 number uses a random train/test split, which lets near-duplicate molecules (same scaffold, different substituent) leak across the split and inflates the score. v2 uses a **Bemis-Murcko scaffold split** — no scaffold appears on both sides — which is the harder, more realistic test of generalizing to genuinely new chemistry. v3 keeps that same scaffold split and tunes each model's hyperparameters with `RandomizedSearchCV`, using scaffold-grouped cross-validation internally so the tuning process is exactly as honest as the final evaluation. v4 swaps the whole classical-ML pipeline for a pretrained chemical transformer (ChemBERTa), fine-tuned on the same scaffold split — no Morgan fingerprint, no RandomForest/XGBoost, just the raw SMILES string tokenized and fed straight to the model. v4 is the current best and the number this project treats as the baseline going forward.

## How it works

The core cheminformatics loop (v1–v3):

1. **Load** the MoleculeNet BBBP dataset (molecules labeled brain-penetrant or not).
2. **Featurize** each molecule's SMILES string into a 2048-bit Morgan fingerprint with RDKit.
3. **Split** — v1 splits randomly; v2 and v3 split by scaffold so train/test share no ring-system cores.
4. **Tune** (v3 only) — `RandomizedSearchCV` over each model's hyperparameters, scored with 5-fold cross-validation grouped by scaffold so no scaffold spans a fold's train and validation side.
5. **Train** a classifier — v1: RandomForest only; v2/v3: RandomForest and XGBoost, head-to-head.
6. **Evaluate** with AUROC on the held-out test set.

v4 replaces steps 2 and 5: instead of a Morgan fingerprint + classical model, it tokenizes the raw SMILES string with a pretrained transformer's own tokenizer and fine-tunes `seyonec/ChemBERTa-zinc-base-v1` (already pretrained on ~2M unlabeled SMILES) directly on the BBBP labels, using the same load/split/evaluate steps as v2/v3.

## Run it

```bash
pip install -r requirements.txt
# v4 also needs the CPU build of torch:
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

python scripts/bbbp_starter.py   # v1: RandomForest, random split
python scripts/bbbp_v2.py        # v2: RandomForest vs. XGBoost, scaffold split
python scripts/bbbp_v3.py        # v3: v2 + hyperparameter tuning
python scripts/bbbp_v4.py        # v4: fine-tuned ChemBERTa
```

## Stack

Python · RDKit · scikit-learn · XGBoost · PyTorch · Transformers (HuggingFace) · pandas · NumPy

## Data

[MoleculeNet BBBP](https://moleculenet.org/) — blood-brain-barrier penetration dataset.

## License

MIT — see [LICENSE](LICENSE).
