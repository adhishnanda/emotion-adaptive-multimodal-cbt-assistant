"""Generate plots and visualizations for IEMOCAP evaluation results."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config.config import PROJECT_ROOT, _load_yaml_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def load_latest_metrics(metrics_dir: Path) -> Optional[Dict]:
    """Load the most recent metrics JSON file."""
    json_files = list(metrics_dir.glob("metrics_*.json"))
    if not json_files:
        logger.warning(f"No metrics files found in {metrics_dir}")
        return None
    
    # Sort by modification time and get latest
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading metrics from {latest_file}")
    
    with open(latest_file, "r") as f:
        return json.load(f)


def plot_confusion_matrices(
    results: Dict,
    class_names: List[str],
    output_dir: Path,
    timestamp: Optional[str] = None,
):
    """Generate confusion matrix plots for each configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_configs = len(results)
    n_cols = 3
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (config_name, metrics) in enumerate(results.items()):
        cm = np.array(metrics["confusion_matrix"])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        ax = axes[idx]
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Normalized Count"},
        )
        ax.set_title(f"Confusion Matrix: {config_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    timestamp_suffix = f"_{timestamp}" if timestamp else ""
    output_path = output_dir / f"confusion_matrices_all{timestamp_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved confusion matrices to {output_path}")
    plt.close()


def plot_individual_confusion_matrix(
    config_name: str,
    cm: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    timestamp: Optional[str] = None,
):
    """Generate individual confusion matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Normalized Count"},
    )
    ax.set_title(f"Confusion Matrix: {config_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    safe_name = config_name.replace("+", "_").replace("-", "_")
    timestamp_suffix = f"_{timestamp}" if timestamp else ""
    output_path = output_dir / f"confusion_matrix_{safe_name}{timestamp_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_macro_f1_comparison(
    results: Dict,
    output_dir: Path,
    timestamp: Optional[str] = None,
):
    """Generate bar chart comparing macro F1 scores across configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = []
    macro_f1_scores = []
    accuracies = []
    
    for config_name, metrics in results.items():
        configs.append(config_name)
        macro_f1_scores.append(metrics["macro_f1"])
        accuracies.append(metrics["accuracy"])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Macro F1 comparison
    colors = sns.color_palette("husl", len(configs))
    bars1 = ax1.bar(range(len(configs)), macro_f1_scores, color=colors)
    ax1.set_xlabel("Configuration", fontsize=12)
    ax1.set_ylabel("Macro F1 Score", fontsize=12)
    ax1.set_title("Macro F1 Score Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha="right")
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars1, macro_f1_scores):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    # Accuracy comparison
    bars2 = ax2.bar(range(len(configs)), accuracies, color=colors)
    ax2.set_xlabel("Configuration", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=45, ha="right")
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, accuracies):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    
    plt.tight_layout()
    
    timestamp_suffix = f"_{timestamp}" if timestamp else ""
    output_path = output_dir / f"metrics_comparison{timestamp_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved metrics comparison to {output_path}")
    plt.close()


def plot_per_class_f1(
    results: Dict,
    class_names: List[str],
    output_dir: Path,
    timestamp: Optional[str] = None,
):
    """Generate heatmap of per-class F1 scores across configurations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build matrix: rows = configurations, columns = classes
    configs = list(results.keys())
    f1_matrix = np.zeros((len(configs), len(class_names)))
    
    for i, config_name in enumerate(configs):
        for j, class_name in enumerate(class_names):
            f1_matrix[i, j] = results[config_name]["per_class"][class_name]["f1"]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(configs) * 0.6)))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=configs,
        ax=ax,
        cbar_kws={"label": "F1 Score"},
    )
    ax.set_title("Per-Class F1 Scores by Configuration", fontsize=14, fontweight="bold")
    ax.set_xlabel("Emotion Class", fontsize=12)
    ax.set_ylabel("Configuration", fontsize=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    timestamp_suffix = f"_{timestamp}" if timestamp else ""
    output_path = output_dir / f"per_class_f1_heatmap{timestamp_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved per-class F1 heatmap to {output_path}")
    plt.close()


def main():
    """Main plotting entry point."""
    # Load config
    from config.config import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg_dict = _load_yaml_config(cfg_path)
    eval_cfg = cfg_dict.get("evaluation", {})
    
    # Directories
    metrics_dir = PROJECT_ROOT / eval_cfg.get("output_dir", "reports/metrics")
    figures_dir = PROJECT_ROOT / eval_cfg.get("figures_dir", "reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Class names
    class_names = eval_cfg.get("class_names", [
        "anger", "happy", "sadness", "neutral", "excited",
        "frustration", "disgust", "fear", "surprise"
    ])
    
    # Load latest metrics
    results = load_latest_metrics(metrics_dir)
    if results is None:
        logger.error("No metrics found. Run evaluation first.")
        sys.exit(1)
    
    # Extract timestamp from filename if needed
    timestamp = None
    
    logger.info("="*60)
    logger.info("Generating Evaluation Plots")
    logger.info("="*60)
    
    # Generate all plots
    plot_confusion_matrices(results, class_names, figures_dir, timestamp)
    
    # Individual confusion matrices
    for config_name, metrics in results.items():
        cm = np.array(metrics["confusion_matrix"])
        plot_individual_confusion_matrix(config_name, cm, class_names, figures_dir, timestamp)
    
    plot_macro_f1_comparison(results, figures_dir, timestamp)
    plot_per_class_f1(results, class_names, figures_dir, timestamp)
    
    logger.info("="*60)
    logger.info("Plot generation complete!")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

