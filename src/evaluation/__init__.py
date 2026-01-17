"""Evaluation package for emotion classification models."""

from .eval_iemocap_multimodal import main as eval_iemocap_main
from .plot_iemocap_results import main as plot_results_main

__all__ = ["eval_iemocap_main", "plot_results_main"]

