# Evaluation Pipeline

This package provides a comprehensive evaluation pipeline for the multimodal CBT assistant, producing thesis-ready metrics and visualizations.

## Overview

The evaluation pipeline evaluates all modality combinations:
- **Unimodal**: Text-only, Audio-only, Video-only
- **Bimodal**: Text+Audio, Text+Video, Audio+Video
- **Trimodal**: Text+Audio+Video (full fusion)

## Usage

### 1. Run Evaluation

```bash
python -m src.evaluation.eval_iemocap_multimodal
```

This will:
- Load all trained models (text, audio, video, fusion)
- Evaluate on the test split of IEMOCAP
- Compute metrics (accuracy, macro F1, per-class precision/recall/F1)
- Save results to `reports/metrics/`

### 2. Generate Plots

```bash
python -m src.evaluation.plot_iemocap_results
```

This will:
- Load the latest evaluation results
- Generate confusion matrices for each configuration
- Create bar charts comparing macro F1 and accuracy
- Generate per-class F1 heatmaps
- Save figures to `reports/figures/`

## Output Files

### Metrics Files (`reports/metrics/`)

- `metrics_YYYYMMDD_HHMMSS.json`: Complete metrics in JSON format
- `summary_YYYYMMDD_HHMMSS.csv`: Summary table with accuracy and F1 scores
- `per_class_metrics_YYYYMMDD_HHMMSS.csv`: Detailed per-class metrics
- `confusion_matrices/`: NumPy arrays of confusion matrices

### Figures (`reports/figures/`)

- `confusion_matrices_all_YYYYMMDD_HHMMSS.png`: All confusion matrices in one figure
- `confusion_matrix_<config>_YYYYMMDD_HHMMSS.png`: Individual confusion matrices
- `metrics_comparison_YYYYMMDD_HHMMSS.png`: Bar charts comparing metrics
- `per_class_f1_heatmap_YYYYMMDD_HHMMSS.png`: Heatmap of per-class F1 scores

## Configuration

Edit `config/config.yaml` under the `evaluation` section:

```yaml
evaluation:
  test_index_path: "data/processed/iemocap_multimodal_index.csv"
  fusion_model_path: "models/fusion/iemocap_multimodal_fusion_best.pt"
  output_dir: "reports/metrics"
  figures_dir: "reports/figures"
  batch_size: 8
  num_workers: 2
  class_names:
    - "anger"
    - "happy"
    - "sadness"
    - "neutral"
    - "excited"
    - "frustration"
    - "disgust"
    - "fear"
    - "surprise"
  # Optional evaluations
  evaluate_daic_woz: false
  evaluate_meld: false
  daic_woz_model_path: "models/audio/daic_woz_best.pt"
  meld_model_path: "models/text/distilbert_meld_best.pt"
```

## Optional Evaluations

### DAIC-WOZ Depression Detection

Set `evaluate_daic_woz: true` to also evaluate the binary depression classification model. This will compute:
- Accuracy
- F1 score
- ROC AUC
- Confusion matrix

Results are saved to `daic_woz_metrics_YYYYMMDD_HHMMSS.json`.

### MELD Text Emotion Classification

Set `evaluate_meld: true` to also evaluate the MELD text emotion model. This will compute:
- Accuracy
- Macro F1
- Per-class metrics

Results are saved to `meld_metrics_YYYYMMDD_HHMMSS.json`.

## Metrics Computed

For each configuration:

1. **Overall Metrics**:
   - Accuracy
   - Macro F1 (unweighted mean of per-class F1)
   - Weighted F1 (support-weighted mean of per-class F1)

2. **Per-Class Metrics**:
   - Precision
   - Recall
   - F1 Score
   - Support (number of samples)

3. **Confusion Matrix**:
   - Normalized confusion matrix for visualization

## Reproducibility

The evaluation uses a fixed random seed (from `config.project.seed`) to ensure reproducible results. All model paths, dataset versions, and timestamps are logged.

## Requirements

- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

All dependencies should already be installed as part of the project requirements.

