# Repository Structure Guide

This document defines the intended journal-facing boundary of the repository without forcing disruptive file moves while active experiments are still in progress.

## Included In The Journal Package

- `data/`
  - Source measurements that back the reported analysis.
  - Each sample directory should keep raw inputs grouped together.
- `experiment/`
  - The current executable analysis layer.
  - Includes notebooks, metric exports, and figure-generation work.
- `outputs/`
  - Curated example outputs suitable for quick inspection or manuscript embedding.

## Explicitly Excluded

- The private sandbox directory at the repository root
  - Reserved for private drafts, scratch work, and non-public iterations.
  - Do not depend on this folder in published notebooks or documentation.

## Migration Target

When the notebooks are stable, the preferred publication layout is:

```text
repo/
|-- data/         # input datasets
|-- notebooks/    # reproducible notebooks only
|-- results/      # generated tables and figures
|-- scripts/      # stable python entry points
|-- docs/         # manuscript-facing documentation
`-- README.md
```

## Suggested Cleanup Order

1. Move notebook files from `experiment/` into `notebooks/`.
2. Move generated PNG, GIF, and CSV artifacts from `experiment/outputs/`, `experiment/plots/`, and `outputs/` into `results/`.
3. Replace notebook-only logic with stable Python scripts in `scripts/`.
4. Update relative paths inside notebooks before removing legacy directories.

## Why This Repo Is Still Conservative

Several files under `experiment/` already have local modifications and untracked outputs.
Because of that, this cleanup pass keeps existing files in place and focuses on documenting a clean public boundary first.
