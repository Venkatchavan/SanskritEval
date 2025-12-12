"""Generate plots for benchmark results."""

import json
import argparse
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_plots")


def plot_sandhi_results(results: list, output_dir: Path):
    """Plot sandhi segmentation results.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory for output plots
    """
    if not results:
        logger.warning("No sandhi results to plot")
        return
    
    models = [r['model'] for r in results]
    metrics = ['precision', 'recall', 'f1', 'exact_match']
    metric_labels = ['Precision', 'Recall', 'F1', 'Exact Match']
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [r[metric] for r in results]
        ax.bar(x + i * width, values, width, label=label)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sandhi Segmentation Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'sandhi_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()


def plot_morphology_results(results: list, output_dir: Path):
    """Plot morphology acceptability results.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory for output plots
    """
    if not results:
        logger.warning("No morphology results to plot")
        return
    
    models = [r['model'] for r in results]
    
    # Overall accuracy plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accuracies = [r['accuracy'] for r in results]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    
    bars = ax.bar(models, accuracies, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Morphological Acceptability - Overall Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add random baseline line
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Random Baseline (50%)')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'morphology_overall.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()
    
    # Phenomenon breakdown plot
    if results and 'by_phenomenon' in results[0]:
        phenomena = list(results[0]['by_phenomenon'].keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(phenomena))
        width = 0.8 / len(models)
        
        for i, result in enumerate(results):
            values = [result['by_phenomenon'][p] for p in phenomena]
            ax.bar(x + i * width, values, width, label=result['model'])
        
        ax.set_xlabel('Phenomenon', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Morphological Acceptability by Phenomenon', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(phenomena)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / 'morphology_by_phenomenon.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()


def plot_combined_comparison(sandhi_results: list, morph_results: list, output_dir: Path):
    """Create combined comparison plot.
    
    Args:
        sandhi_results: Sandhi evaluation results
        morph_results: Morphology evaluation results
        output_dir: Directory for output plots
    """
    # Get all unique models
    all_models = set()
    if sandhi_results:
        all_models.update(r['model'] for r in sandhi_results)
    if morph_results:
        all_models.update(r['model'] for r in morph_results)
    
    models = sorted(all_models)
    
    if not models:
        logger.warning("No models to plot")
        return
    
    # Create lookup dicts
    sandhi_dict = {r['model']: r for r in sandhi_results}
    morph_dict = {r['model']: r for r in morph_results}
    
    # Prepare data
    sandhi_f1 = [sandhi_dict.get(m, {}).get('f1', 0) for m in models]
    morph_acc = [morph_dict.get(m, {}).get('accuracy', 0) for m in models]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Sandhi F1
    colors1 = plt.cm.Blues(np.linspace(0.4, 0.8, len(models)))
    bars1 = ax1.bar(models, sandhi_f1, color=colors1)
    
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('Task A: Sandhi Segmentation', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Morphology Accuracy
    colors2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(models)))
    bars2 = ax2.bar(models, morph_acc, color=colors2)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)
    
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Random')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Task B: Morphological Acceptability', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    plt.suptitle('SanskritEval Benchmark Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'combined_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()


def main():
    if not PLOTTING_AVAILABLE:
        logger.error("matplotlib not available. Install with: pip install matplotlib")
        return
    
    parser = argparse.ArgumentParser(description='Generate plots from benchmark results')
    parser.add_argument(
        '--sandhi-results',
        type=Path,
        default=Path('results/sandhi_results.json'),
        help='Path to sandhi results JSON'
    )
    parser.add_argument(
        '--morph-results',
        type=Path,
        default=Path('results/morphology_results.json'),
        help='Path to morphology results JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/plots'),
        help='Directory for output plots'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    sandhi_results = []
    morph_results = []
    
    if args.sandhi_results.exists():
        with open(args.sandhi_results, 'r') as f:
            sandhi_results = json.load(f)
        logger.info(f"Loaded {len(sandhi_results)} sandhi results")
    else:
        logger.warning(f"Sandhi results not found: {args.sandhi_results}")
    
    if args.morph_results.exists():
        with open(args.morph_results, 'r') as f:
            morph_results = json.load(f)
        logger.info(f"Loaded {len(morph_results)} morphology results")
    else:
        logger.warning(f"Morphology results not found: {args.morph_results}")
    
    # Generate plots
    if sandhi_results:
        plot_sandhi_results(sandhi_results, args.output_dir)
    
    if morph_results:
        plot_morphology_results(morph_results, args.output_dir)
    
    if sandhi_results or morph_results:
        plot_combined_comparison(sandhi_results, morph_results, args.output_dir)
    
    logger.info(f"\nAll plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
