"""Metrics for sandhi segmentation evaluation."""

from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class SandhiMetrics:
    """Metrics for sandhi segmentation task."""
    precision: float
    recall: float
    f1: float
    exact_match: float
    total_examples: int
    true_positives: int
    false_positives: int
    false_negatives: int


def extract_boundaries(text: str) -> set:
    """Extract boundary positions from segmented text.
    
    Boundaries are marked by spaces or '+'.
    
    Args:
        text: Segmented text like "धर्म + क्षेत्रे"
    
    Returns:
        Set of boundary character positions
    """
    boundaries = set()
    pos = 0
    for char in text:
        if char in (' ', '+'):
            boundaries.add(pos)
        else:
            pos += 1
    return boundaries


def normalize_for_comparison(text: str) -> str:
    """Remove spaces and boundary markers for text comparison.
    
    Args:
        text: Text with possible spaces/markers
        
    Returns:
        Normalized text without spaces/markers
    """
    return re.sub(r'[\s+]', '', text)


def compute_sandhi_metrics(
    predictions: List[Tuple[str, str]],
    references: List[Tuple[str, str]]
) -> SandhiMetrics:
    """Compute precision, recall, F1 for sandhi segmentation.
    
    Args:
        predictions: List of (fused, predicted_segmented) tuples
        references: List of (fused, gold_segmented) tuples
        
    Returns:
        SandhiMetrics with P/R/F1 for boundary detection
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_matches = 0
    
    for (fused_pred, pred_seg), (fused_ref, ref_seg) in zip(predictions, references):
        # Verify same fused form
        if normalize_for_comparison(fused_pred) != normalize_for_comparison(fused_ref):
            raise ValueError(
                f"Fused forms don't match: '{fused_pred}' vs '{fused_ref}'"
            )
        
        # Extract boundary positions
        pred_boundaries = extract_boundaries(pred_seg)
        ref_boundaries = extract_boundaries(ref_seg)
        
        # Compute TP/FP/FN for this example
        tp = len(pred_boundaries & ref_boundaries)
        fp = len(pred_boundaries - ref_boundaries)
        fn = len(ref_boundaries - pred_boundaries)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Exact match if all boundaries match
        if pred_boundaries == ref_boundaries:
            exact_matches += 1
    
    # Compute metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = exact_matches / len(predictions) if predictions else 0.0
    
    return SandhiMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=exact_match,
        total_examples=len(predictions),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn
    )


def format_metrics(metrics: SandhiMetrics) -> str:
    """Format metrics for display.
    
    Args:
        metrics: Computed metrics
        
    Returns:
        Formatted string
    """
    return f"""Sandhi Segmentation Metrics:
  Precision:    {metrics.precision:.3f} ({metrics.true_positives} / {metrics.true_positives + metrics.false_positives})
  Recall:       {metrics.recall:.3f} ({metrics.true_positives} / {metrics.true_positives + metrics.false_negatives})
  F1 Score:     {metrics.f1:.3f}
  Exact Match:  {metrics.exact_match:.3f} ({int(metrics.exact_match * metrics.total_examples)} / {metrics.total_examples})
"""
