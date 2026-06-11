# Quantum Image Encoding (QIE) for Biosignal Classification

Code repository for the paper:

> **Preliminary Analysis of Quantum Image Encoding for Enhanced Biosignal Classification**
> J. Hejda, M. Sokol, P. Volf, P. Kutilek
> Czech Technical University in Prague, Faculty of Biomedical Engineering

## Overview

This repository contains the source code for reproducing the experiments described in the paper. We investigate Quantum Image Encoding (QIE), a quantum-circuit-inspired analogue of the Gramian Angular Field (GAF), for converting 1D biosignals into 2D images for CNN/SVM classification. The study includes:

- Classical GAF baselines (GASF, GADF) and the proposed QIE encoding
- Two QIE circuit extensions (QIE-NLP, QIE-Ent)
- Benchmark across eight public biosignal datasets under subject-grouped cross-validation
- End-to-end hardware validation on IBM Quantum Heron r3 (ibm_boston)

## Repository Structure

```
src/
  transformations.py      # All 1D-to-2D encoding implementations (GAF, QIE, etc.)
  novel_transforms.py     # Experimental transforms (ASPAG, MSGK, PSSAM)
  cnn_model.py            # SimpleCNN architecture used in CNN benchmarks
experiments/
  run_unified_benchmark.py    # PCA + RandomForest benchmark across datasets
  run_cnn_benchmark.py        # CNN benchmark across datasets
  run_experiment.py           # WESAD encoding comparison (Table I)
configs/
  default.yaml            # Default experiment configuration
```

## Requirements

```
numpy
scipy
scikit-learn
pyts
torch (CUDA build recommended for CNN experiments)
matplotlib
```

Install:
```bash
pip install numpy scipy scikit-learn pyts torch matplotlib pyyaml
```

## Data

This repository does not include datasets. The following public datasets are used:

| Dataset | Source | Signal |
|---------|--------|--------|
| WESAD | [Schmidt et al., 2018](https://doi.org/10.1145/3242969.3242985) | ECG |
| CLARE | Cognitive Load Assessment in REaltime | ECG |
| CLAS | Cognitive Load Assessment Study | ECG |
| StressID | Stress Identification Dataset | ECG |
| CASE | Continuously Annotated Signals of Emotion | ECG |
| DREAMER | [Katsigiannis & Ramzan, 2018](https://doi.org/10.1109/JBHI.2017.2688239) | ECG |
| PPG FieldStudy | PPG-DaLiA Dataset | ECG |
| SWELL | [Koldijk et al., 2014](https://doi.org/10.1145/2663204.2663257) | ECG |

Preprocessed data (`.pkl` files) should be placed in a `data/processed/` directory with filenames like `WESAD_30s_50ovl.pkl`. Each pickle contains keys: `data` (N, window_samples, channels), `labels` (N,), `participants` (N,).

## Usage

### Encoding comparison (Table I)

```bash
python experiments/run_experiment.py --dataset WESAD --output results/
```

### CNN benchmark across datasets

```bash
python experiments/run_cnn_benchmark.py --data-dir data/processed/ --output results/cnn_benchmark.json
```

### Unified RF benchmark

```bash
python experiments/run_unified_benchmark.py --data-dir data/processed/ --output results/unified_benchmark.json
```

## QIE Encoding

The core QIE encoding builds each pixel of a 32x32 image from a single-qubit circuit:

```
QIE_{ij} = <Z> of R_y(2*phi_j) R_y(2*phi_i) |0>  =  cos(2*(phi_i + phi_j))
```

where `phi_k = arccos(s_k)` and `s_k` are PAA-reduced, [-1,1]-scaled time-series values.

Two extensions are also provided:
- **QIE-NLP**: `R_y(pi*a*b) R_z((a-b)^2)` with nonlinear cross-term
- **QIE-Ent**: `R_y(pi*a) R_z(pi*a*b) R_y(pi*b)` with entanglement-inspired product interaction

See `src/transformations.py` for implementations.

## Hardware Execution

Hardware experiments use IBM Quantum's `ibm_boston` (Heron r3, 156 qubits) via `qiskit-ibm-runtime`. The qubit-parallel scheme packs ~100 independent single-qubit circuits per job, achieving ~1 QPU-second per image. Hardware scripts are not included in this repository due to IBM API credential requirements.

## Related

- [QTS2D](https://doi.org/10.1016/j.softx.2025.102327) -- our quantum time-series-to-image encoding library (SoftwareX, 2025)

## Citation

```bibtex
@inproceedings{hejda2026qie,
  author    = {Hejda, Jan and Sokol, Marek and Volf, Petr and Kut\'ilek, Patrik},
  title     = {Preliminary Analysis of Quantum Image Encoding for Enhanced Biosignal Classification},
  year      = {2026},
}
```

## License

MIT
