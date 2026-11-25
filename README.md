# CRISPR-AI: Safe-Edit Off-Target Prediction

Deep learning models that predict CRISPR/Cas9 off-target cleavage from sgRNA-DNA
sequence pairs, developed for an MSc project. The pipeline trains and compares
GRU, BiLSTM, CNN-BiLSTM and CNN-BiLSTM-Attention models, tunes them with random
search and a genetic algorithm, evaluates them on a held-out test set, and
analyses what they learn using interpretability methods.

## Data

The models use the DeepCRISPR off-target classification data (HEK293T and K562
cell lines). The data is **not included** in this repository. To reproduce:

1. Obtain the DeepCRISPR release and place `paper_data-classification.tar.gz`
   in a `DeepCRISPR-master/` folder at the repository root.
2. Run `src/extract_deepcrispr.py` to extract it into `data/`.

Each sgRNA-DNA pair (23 nt) is one-hot encoded to a 4-channel array of shape
`(23, 4)`.

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
| 8 | `step8_interpretability_analysis.py`, `step8_enhanced_figures.py` | Integrated gradients, in-silico mutagenesis, saliency, CNN filters |
| 9 | `step9_performance_final_figures.py` | Performance figures |
| 11 | `step11_benchmark_comparison.py` | Comparison with published tools |

Note on the 5-channel run: `step4c_true_ga_4MODELS.py` contains a fourth run,
`CNN-BiLSTM-5ch`, that loads `data/X_train_5ch.npy` and `data/X_val_5ch.npy`.
No script here generates those arrays and no 5-channel results are included,
so every result reported below uses the 4-channel encoding.

## Results

Two separate test-set evaluations are included. They are not directly
comparable — they differ in both the hyperparameters and the training
protocol — so both are reported rather than one being treated as superseding
the other.

### Final evaluation: GA-tuned models, Option B protocol

Produced by `step5_final_test_optionB.py`
(`results/test_evaluation/final_test_results_optionB.csv`). Hyperparameters
come from the genetic-algorithm search; each model is then retrained on the
combined train+validation set (90% of the data, with an internal 10% split for
early stopping) and evaluated once on the held-out test set.

| Model | Test AUPRC | Test AUROC |
| --- | --- | --- |
| GRU-4ch | 0.704 | 0.993 |
| BiLSTM-4ch | 0.710 | 0.991 |
| CNN-BiLSTM-4ch | 0.750 | 0.994 |

### Earlier evaluation: pre-GA, train-split only

`results/final_training/test_set_performance.csv`, covering four
architectures with full classification metrics. These models were trained on
the training split alone with untuned hyperparameters, before the GA search.

| Model | AUPRC | AUROC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| GRU | 0.711 | 0.994 | 0.422 | 0.818 | 0.557 |
| BiLSTM | 0.386 | 0.981 | 0.476 | 0.303 | 0.370 |
| CNN-BiLSTM | 0.621 | 0.992 | 0.347 | 0.788 | 0.481 |
| CNN-BiLSTM-Attention | 0.657 | 0.992 | 0.173 | 0.879 | 0.289 |

BiLSTM is the clearest difference between the two tables (0.386 to 0.710).
Both the GA-tuned hyperparameters and the extra training data in the Option B
protocol change between the runs, so this repository does not attribute the
gain to either on its own.

### Other result files

- `results/random_search/` — random hyperparameter search scores
- `results/random_search_test/` — test metrics for random-search models, which
  performed poorly (AUPRC 0.03–0.55) and were not used for the final models
- `results/true_ga/` — per-generation GA progress and final population
- `results/training_summaries/` — training/validation loss and runtime per
  architecture (loss curves, not test metrics)
- `results/interpretability/` — integrated gradients, in-silico mutagenesis,
  saliency and CNN filter analyses

Figures are in `figures/`, numbered in roughly the order they appear in the
write-up.

## Setup

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
