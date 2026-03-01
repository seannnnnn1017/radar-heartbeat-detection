# Radar Heartbeat Detection

This repository is organized as the journal-facing version of the mmWave heartbeat detection project.
The private sandbox directory at the repository root is intentionally excluded from the documented scope and should be treated as exploratory workspace only.

## Scope

The repository currently preserves three public-facing areas:

- `data/`: paired mmWave and ECG samples used for analysis.
- `experiment/`: notebooks and evaluation artifacts that reproduce the current analysis flow.
- `outputs/`: representative exported figures and CSV summaries.

The private sandbox directory is not part of the journal package and should not be referenced by reproducible workflows, figures, or documentation.

## Recommended Use

1. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate mmv-env
```

2. Use the notebooks in `experiment/` as the current source of truth for analysis and figure generation.

3. Treat files in `outputs/` as paper-ready examples, not the complete experiment log.

## Repository Layout

```text
radar-heartbeat-detection/
|-- data/                # raw and paired measurement samples
|-- experiment/          # reproducible notebooks and evaluation exports
|-- outputs/             # selected generated figures / summary csv files
|-- docs/                # repository conventions for the journal version
|-- environment.yml      # conda environment definition
|-- .gitignore
`-- README.md
```

Detailed cleanup rules and the target journal-facing structure are documented in `docs/REPO_STRUCTURE.md`.

## Current Notes

- `experiment/starandard_process_alldata.py` is currently empty; the notebooks are the active implementation path.
- Generated files and intermediate experiment assets still exist inside `experiment/`; they are preserved for now to avoid breaking in-progress work.
- If you later want a stricter publication package, the next step is to split `experiment/` into `notebooks/`, `results/`, and `scripts/` after notebook paths are updated.
