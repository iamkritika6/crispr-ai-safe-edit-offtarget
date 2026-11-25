# CRISPR-AI: Safe-Edit Off-Target Prediction

Deep learning models that predict CRISPR/Cas9 off-target cleavage from
sgRNA-DNA sequence pairs, developed for an MSc project (University of
Huddersfield). Seven architectures were screened; GRU, BiLSTM and CNN-BiLSTM
were tuned with random search and a genetic algorithm (CNN-BiLSTM-Attention was
dropped before the genetic algorithm) and evaluated on a held-out test set, then
analysed with interpretability methods. Builds on DeepCRISPR (Chuai et al.,
2018) and the evaluation protocol of Toufikuzzaman et al. (2024).
Author: Kritika Singh Chauhan.

## Data

The models use the DeepCRISPR off-target classification data (HEK293T and K562
cell lines). The data is **not included** in this repository. To reproduce:

1. Obtain the DeepCRISPR release and place `paper_data-classification.tar.gz`
   in a `DeepCRISPR-master/` folder at the repository root.
2. Run `src/extract_deepcrispr.py` to extract it into `data/`.

Each sgRNA-DNA pair (23 nt) is one-hot encoded to a 4-channel array of shape
`(23, 4)`.

153,233 sgRNA-DNA pairs, 656 positives (about 1:232 imbalance), split 80/10/10
and stratified by class and cell line. The test set contains 66 positives.

## Pipeline

Run from the repository root, in this order:

| Step | Script | Purpose |
| --- | --- | --- |
| 0 | `extract_deepcrispr.py`, `load_deepcrispr_data.py` | Extract and inspect the raw files |
| 1 | `step1_data_preprocessing.py` | Parse and clean HEK293T + K562 data |
| 2 | `step2_encoding.py` | One-hot encoding, train/val/test split |
| 3 | `step3_build_models.py` | Model architectures |
| 4 | `step4b_random_search.py` | Random hyperparameter search |
| 4 | `step4c_true_ga_4MODELS.py` | Genetic-algorithm search (crossover + mutation) |
| 4 | `step4c_bilstm_memory_optimized.py`, `step4c_cnn_bilstm_4ch_optimized.py` | Memory-optimised GA runs |
| 5 | `step5_final_test_optionB.py` | Final test-set evaluation |
| 6 | `step6_random_search_test.py` | Test-set evaluation of random-search models |
| 8 | `step8_interpretability_analysis.py`, `step8_enhanced_figures.py` | Integrated gradients, CNN filter analysis (information content), in-silico mutagenesis (seed vs distal, positional sensitivity), biological alignment scoring |
| 9 | `step9_performance_final_figures.py` | Performance figures |
| 11 | `step11_benchmark_comparison.py` | Comparison against values reported in the literature; not a like-for-like benchmark |

Note on the 5-channel run: `step4c_true_ga_4MODELS.py` contains a fourth run,
`CNN-BiLSTM-5ch`, that loads `data/X_train_5ch.npy` and `data/X_val_5ch.npy`.
No script here generates those arrays and no 5-channel results are included,
so every result reported below uses the 4-channel encoding.

## Results

Produced by `step5_final_test_optionB.py`
(`results/test_evaluation/final_test_results_optionB.csv`). Hyperparameters
come from the genetic-algorithm search; each model is then retrained on the
combined train+validation set (137,909 pairs, 90% of the data), with an
internal 90/10 split of that data for early-stopping monitoring, and evaluated
once on the held-out test set.

| Model | Test AUPRC |
| --- | --- |
| GRU-4ch | 0.7040 |
| BiLSTM-4ch | 0.7097 |
| CNN-BiLSTM-4ch | 0.7497 |

### Other result files

- `results/random_search/` — random hyperparameter search scores
- `results/random_search_test/` — test metrics for random-search models, which
  performed poorly (AUPRC 0.03–0.55) and were not used for the final models
- `results/true_ga/` — per-generation GA progress and final population
- `results/training_summaries/` — baseline training/validation loss and runtime
  per architecture (loss curves, not test metrics)
- `results/interpretability/` — integrated gradients, in-silico mutagenesis
  (positional sensitivity, mismatch types), CNN filter analyses and
  `biological_alignment_score.txt` (GRU 60%, BiLSTM 60%, CNN-BiLSTM 80%),
  all written by `step8_interpretability_analysis.py`

Figures are in `figures/`, numbered in roughly the order they appear in the
write-up.

## Limitations

- The test set contains only 66 positive examples, so all reported metrics
  carry wide uncertainty and small differences between models should not be
  read as meaningful.
- The thesis reports poor cross-dataset generalisation on GUIDE-seq; that
  evaluation is not reproduced in this repository.
- The baseline training script that produced
  `results/training_summaries/baseline_training_summary.csv` is not included.
- All results come from a single dataset (DeepCRISPR HEK293T and K562) and have
  not been replicated on independent data.

## Setup

Developed with Python 3.12.8 and PyTorch 2.7.1 (CUDA 11.8 build).

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended; the scripts fall back to CPU.

## Repository layout

```
src/        pipeline scripts
results/    summary metrics (CSV)
figures/    thesis figures (PNG)
```

Large artefacts (raw data, encoded arrays, trained weights, checkpoints,
reference papers) are excluded by `.gitignore`.
