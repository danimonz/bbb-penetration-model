# CLAUDE.md — Drug Discovery Project

Instructions for Claude Code working in this folder. Keep this file updated as the project evolves.

---

## North Star

Learn computational drug discovery by **building and deeply understanding every line** — not vibe-coding — toward becoming someone who can architect *and explain* ML models for drug discovery.

- **Destination:** the San Francisco Bay Area computational drug-discovery ecosystem (UCSF, Genesis Therapeutics, insitro, Genentech). Long-game — orient toward it, don't chase it now.
- **Guiding principle:** understanding > output. Foundation first. Learn as we go.

## Current State  *(update this section as it changes)*

- **Baseline model:** `scripts/bbbp_starter.py` — RDKit Morgan fingerprints (ECFP4, radius 2, 2048 bits) + RandomForest on the MoleculeNet BBBP dataset. 0.93 test AUROC on a **random** split (kept as historical baseline only — inflated, see below).
- **Current best (own-split) model:** `scripts/bbbp_v4.py` — fine-tuned ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`), same scaffold split as v2/v3. **0.912 AUROC**, a single run. Reasoning logged in `drug-discovery-wiki/decisions/2026-09-02-chemberta-fine-tune.md`.
- **Official TDC leaderboard result:** `scripts/bbbp_tdc.py` — same fine-tuning approach, but on TDC's own `BBB_Martins` scaffold split, averaged over TDC's required 5 seeds (each seed keeps its best epoch by validation AUROC rather than a fixed 3). **0.856 ± 0.021 AUROC** — ~rank 19/27 on the public leaderboard (top: MiniMol 0.924). This, not 0.912, is the honest number to quote for leaderboard comparisons — see `drug-discovery-wiki/decisions/2026-09-03-tdc-leaderboard-entry.md` for why the two numbers differ. Not yet submitted to TDC's public leaderboard (that's a separate, not-yet-made decision).
- `scripts/bbbp_v3.py` (tuned RandomForest 0.861 / XGBoost 0.835 AUROC, `RandomizedSearchCV` + scaffold-grouped CV) kept as the best classical-ML result. `scripts/bbbp_v2.py` (untuned RF vs. XGBoost on scaffold split) kept as the pre-tuning baseline. Reasoning for v2/v3 logged in `drug-discovery-wiki/decisions/2026-08-28-scaffold-split-over-random-split.md` and `drug-discovery-wiki/decisions/2026-09-01-grouped-cv-for-tuning.md`.
- v2, v3, and v4 pushed to GitHub (danimonz/bbb-penetration-model); the TDC script not yet pushed.
- **Environment:** Ubuntu. Main work in a `.venv` (Python 3.12), pinned in `requirements.txt` — includes `torch` (CPU build) and `transformers` for v4. A **separate `.venv-tdc` (Python 3.11, gitignored)** is used for PyTDC (3.12 breaks its install — old pinned scikit-learn has no cp312 wheel and fails to build from source without a compiler); installed via the `deadsnakes` PPA plus `build-essential`.
- **The ladder (improvement path):** RandomForest baseline → XGBoost + **scaffold split** (done) → tune (done) → fine-tune a chemical transformer (done — ChemBERTa 0.912 AUROC on our split) → official **TDC leaderboard** entry (**done, currently here** — 0.856 ± 0.021 AUROC on TDC's split/harness; public submission not yet decided). A graph neural net remains an optional side-branch, not required for the ladder to continue.

## How Claude Code Should Work Here

1. **Teach, don't just do.** When writing or changing code, explain what each part does and *why*, at the level of someone who wants to truly understand it. Clarity over cleverness.
2. **No black boxes.** Introducing a new library, function, or concept? Define it in plain English and create/point to a wiki note for it (see below).
3. **Understand-first.** Explain the plan before a big change; summarize what changed and why after.
4. **Honest evaluation.** Use **scaffold splits** for any leaderboard-relevant number — never let a random split flatter results. Report metrics honestly, including drops.
5. **Reproducibility.** Keep code runnable, commented, and seeded. Maintain `requirements.txt`. Keep `.gitignore` hygiene — never commit `.venv/`, `data/`, or `__pycache__/`.
6. **One rung at a time.** Establish a baseline before adding complexity. Finish the current rep before starting the next.
7. **Maintain the wiki.** When we learn a concept or make a decision, create/update the relevant atomic note and link it.
8. **Summarize notes.** When I type or paste daily notes, produce a concise summary and flag 1–3 concepts worth promoting to permanent wiki notes.

## Folder Structure

```
drug-discovery/
├── CLAUDE.md              # this file
├── README.md
├── LICENSE                # MIT
├── requirements.txt       # pinned deps — pip install -r requirements.txt
├── scripts/               # code (bbbp_starter.py, bbbp_v2.py, ...)
├── data/                  # datasets (gitignored) — created if/when we cache data locally
├── models/                # saved trained models (gitignore if large) — created if/when we persist a model
└── drug-discovery-wiki/   # the Obsidian vault (gitignored — personal, not pushed to GitHub)
    ├── 00-index.md        # map of content (MOC) — the wiki's front door
    ├── daily/             # daily notes (journal / fleeting capture)
    ├── concepts/          # atomic permanent notes — ONE idea per note
    ├── projects/          # per-project notes (BBB model, leaderboard run, ...)
    ├── resources/         # reading list, papers, links
    └── decisions/         # key choices + the reasoning behind them (running log)
```

`data/` and `models/` don't exist yet — created the first time we actually cache a dataset or persist a trained model, per the "let structure grow from what we encounter" guardrail.

## Wiki Conventions (Zettelkasten)

- **Atomic notes.** One concept per note in `concepts/` (e.g., `auroc.md`, `morgan-fingerprint.md`, `random-forest.md`, `scaffold-split.md`, `fine-tuning.md`).
- **Each concept note holds:** a plain-English definition, *why it matters for this project*, *how it shows up in the code*, and `[[links]]` to related notes.
- **Link liberally.** The graph of connections is where the value is.
- **Daily notes are fleeting** — fast capture, not the wiki itself. The wiki *emerges* when daily notes are distilled into atomic concept notes.
- **`00-index.md` is the map** — links to the main concept clusters and active projects. Update it weekly.
- **Tags** for status/topic: `#concept`, `#project`, `#todo`.

## Daily Workflow

1. **During the day:** capture on paper (your natural mode) + jot quick items in the Obsidian daily note.
2. **End of day:** type a short summary of the paper journal into that day's daily note.
3. **Distill:** ask Claude Code to summarize the daily note and extract 1–3 concepts worth promoting.
4. **Promote:** turn each concept into (or update) an atomic note in `concepts/`, linked from the daily note and the index. *This step is what turns a journal into a wiki.*
5. **Weekly:** skim `concepts/`, tidy links, update `00-index.md`.

## Division of Labor

- **Claude Code (terminal):** hands on code and files — writing/editing scripts, running experiments, maintaining wiki files, summarizing notes.
- **The chat:** the architect layer — concepts, the "why," planning the ladder, Bay Area steering, deciding what to build next.

## Personal Guardrails

- One rung at a time; finish the current rep before starting the next thing.
- Resources are just-in-time references, not a curriculum to complete before building.
- Keep the system **lean** — a wiki you actually maintain beats an elaborate one you abandon. Let structure grow from what you encounter, not upfront.
