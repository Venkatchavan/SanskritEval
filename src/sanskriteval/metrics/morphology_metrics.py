"""Metrics for morphological acceptability evaluation."""

from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict


@dataclass
class MorphologyMetrics:
    """Metrics for morphological acceptability task."""
    accuracy: float
    total_pairs: int
    correct_pairs: int
    by_phenomenon: Dict[str, float]  # accuracy per phenomenon (case/number)
    by_stem_class: Dict[str, float]  # accuracy per declension


def compute_morphology_metrics(
    scores: List[Tuple[float, float]],
    phenomena: List[str],
    stem_classes: List[str]
) -> MorphologyMetrics:
    """Compute acceptability accuracy for morphological contrast sets.
    
    Model is correct if: score(grammatical) > score(ungrammatical)
    
    Args:
        scores: List of (gram_score, ungram_score) tuples
        phenomena: List of phenomenon labels ('case' or 'number')
        stem_classes: List of stem class labels ('a-stem', etc.)
        
    Returns:
        MorphologyMetrics with overall and per-phenomenon accuracy
    """
    if not (len(scores) == len(phenomena) == len(stem_classes)):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores, "
            f"{len(phenomena)} phenomena, {len(stem_classes)} stem_classes"
        )
    
    correct = 0
    by_phenomenon = defaultdict(lambda: {'correct': 0, 'total': 0})
    by_stem = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for (gram_score, ungram_score), phenom, stem in zip(scores, phenomena, stem_classes):
        is_correct = gram_score > ungram_score
        
        if is_correct:
            correct += 1
            by_phenomenon[phenom]['correct'] += 1
            by_stem[stem]['correct'] += 1
        
        by_phenomenon[phenom]['total'] += 1
        by_stem[stem]['total'] += 1
    
    # Compute accuracies
    accuracy = correct / len(scores) if scores else 0.0
    
    phenomenon_acc = {
        phenom: stats['correct'] / stats['total']
        for phenom, stats in by_phenomenon.items()
    }
    
    stem_acc = {
        stem: stats['correct'] / stats['total']
        for stem, stats in by_stem.items()
    }
    
    return MorphologyMetrics(
        accuracy=accuracy,
        total_pairs=len(scores),
        correct_pairs=correct,
        by_phenomenon=phenomenon_acc,
        by_stem_class=stem_acc
    )


def format_metrics(metrics: MorphologyMetrics) -> str:
    """Format metrics for display.
    
    Args:
        metrics: Computed metrics
        
    Returns:
        Formatted string
    """
    lines = [
        "Morphological Acceptability Metrics:",
        f"  Overall Accuracy: {metrics.accuracy:.3f} ({metrics.correct_pairs} / {metrics.total_pairs})",
        "",
        "  By Phenomenon:",
    ]
    
    for phenom, acc in sorted(metrics.by_phenomenon.items()):
        lines.append(f"    {phenom:10s}: {acc:.3f}")
    
    if metrics.by_stem_class:
        lines.append("")
        lines.append("  By Stem Class:")
        for stem, acc in sorted(metrics.by_stem_class.items()):
            lines.append(f"    {stem:15s}: {acc:.3f}")
    
    return "\n".join(lines)
