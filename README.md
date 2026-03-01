# Millimeter-Wave Radar Heart Rate Monitoring via Time-Frequency Fusion and Composite Filtering with ECG Alignment

Chi Hung Wang1*, Xiang Shun Yang1, Yu Hsiang Hsiang1

1 Dept. of Artificial Intelligence Technology and Application, Feng Chia University, Taichung, Taiwan
*Corresponding author: Chi-Hung Wang (email: chihwang@o365.fcu.edu.tw)

## Abstract

Non-contact heart rate monitoring is becoming increasingly important for telecare and clinical infection control. Existing methods such as remote photoplethysmography (rPPG) and infrared thermography (IRT) can estimate heart rate from optical and thermal spectral features, but rPPG is highly sensitive to illumination changes, while IRT is vulnerable to airflow and temperature fluctuations, limiting their use in long-term and clinical scenarios. In contrast, millimeter-wave radar offers strong resistance to lighting interference and low-cost deployment, enabling heartbeat estimation from subtle chest motion. However, its performance can still be degraded by respiratory drift, reflection clutter, and phase instability, which require further suppression and compensation. To address these issues, this study proposes a time-frequency fusion framework that combines composite filtering and cepstral analysis to improve the stability and physiological consistency of millimeter-wave heart rate estimation. The system applies layered denoising with band-pass (BP), wavelet, median, and amplification (AMP) filters, followed by drift compensation using Peaks + Drift and electrocardiogram (ECG) alignment. Fast Fourier transform (FFT) and cepstrum-based reconstruction are then used to recover periodic structure, and the dominant peak is tracked within the physiological range to estimate heart rate. To evaluate robustness, we further conduct comparisons across composite filtering sequences, analyses of time-frequency parameters, and stress tests under multiple noise conditions. Experimental results show that the AMP to BP sequence performs best among twelve tested combinations. With a 4-second window, a 0.5-second hop, and q_min = 0.5, the proposed method achieves a mean absolute error (MAE) of 4.3 bpm, a root mean square error (RMSE) of 8.4 bpm, and a bias of 0.8 bpm. In addition, stress tests using six noise types at signal-to-noise ratios (SNRs) from -5 to 20 dB show that the filtered heart rate estimates decrease steadily with increasing SNR in all scenarios and remain consistently lower than the raw estimates. The overall estimation behavior is continuous and reproducible, indicating that this framework can provide stable, low-cost non-contact heart rate monitoring and may serve as a practical reference for future clinical applications. The project associated with this work is publicly available at https://github.com/seannnnnn1017/radar-heartbeat-detection.

This repository contains the mmWave heartbeat detection project, including data, analysis notebooks, and selected output artifacts.
The private sandbox directory at the repository root is intentionally excluded from the main project scope and should be treated as exploratory workspace only.

## Scope

The repository currently preserves three public-facing areas:

- `data/`: 182 sample folders, each containing paired mmWave (`mmv.csv`) and ECG (`1_1lead.csv`) signals used for analysis.
- `experiment/`: notebooks and evaluation artifacts that reproduce the current analysis flow.
- `outputs/`: representative exported figures and CSV summaries.

The private sandbox directory is not part of the main repository workflow and should not be referenced by reproducible workflows, figures, or documentation.

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
|-- docs/                # repository conventions and cleanup notes
|-- environment.yml      # conda environment definition
|-- .gitignore
`-- README.md
```

Detailed cleanup rules and the target repository structure are documented in `docs/REPO_STRUCTURE.md`.

## Current Notes

- `experiment/starandard_process_alldata.py` is currently empty; the notebooks are the active implementation path.
- Generated files and intermediate experiment assets still exist inside `experiment/`; they are preserved for now to avoid breaking in-progress work.
- If you later want a stricter publication package, the next step is to split `experiment/` into `notebooks/`, `results/`, and `scripts/` after notebook paths are updated.
