# BBB Penetration Model

A machine-learning model that predicts whether a molecule crosses the **blood-brain barrier (BBB)** from its chemical structure alone — a real filter used in early drug design to judge whether a compound can reach the brain.

## Result

Trained a Random Forest on 2,039 molecules (1,631 train / 408 test).

**Test AUROC: 0.93**

## How it works

The core cheminformatics loop:

1. **Load** the MoleculeNet BBBP dataset (molecules labeled brain-penetrant or not).
2. **Featurize** each molecule's SMILES string into a 2048-bit Morgan fingerprint with RDKit.
3. **Train** a Random Forest classifier.
4. **Evaluate** with AUROC on a held-out test set.

## Run it

```bash
pip install pandas numpy rdkit scikit-learn
python bbbp_starter.py
```

## Stack

Python · RDKit · scikit-learn · pandas · NumPy

## Data

[MoleculeNet BBBP](https://moleculenet.org/) — blood-brain-barrier penetration dataset.
 
