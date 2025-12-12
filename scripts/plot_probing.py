"""Generate layer-wise probing plots."""

import json
import argparse
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("plot_probing")


def plot_layer_curves(results: list, output_dir: Path):
    """Plot accuracy/F1 curves across layers.
    
    Args:
        results: List of probing results
        output_dir: Directory for output plots
    """
    if not results:
        logger.warning("No results to plot")
        return
    
    # Create figure with subplots
    n_tasks = len(results)
    fig, axes = plt.subplots(1, n_tasks, figsize=(7*n_tasks, 5))
    
    if n_tasks == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        # Extract data
        layers = [r['layer'] for r in result['results']]
        accuracies = [r['accuracy'] for r in result['results']]
        f1_scores = [r['f1_score'] for r in result['results']]
        
        # Plot curves
        ax.plot(layers, accuracies, 'o-', linewidth=2, markersize=6, 
                label='Accuracy', color='#2E86AB')
        ax.plot(layers, f1_scores, 's-', linewidth=2, markersize=6,
                label='F1 Score', color='#A23B72')
        
        # Find and mark best layer
        best_idx = np.argmax(f1_scores)
        best_layer = layers[best_idx]
        best_f1 = f1_scores[best_idx]
        
        ax.axvline(x=best_layer, color='red', linestyle='--', alpha=0.5, linewidth=2)
        ax.annotate(f'Peak: Layer {best_layer}\nF1={best_f1:.3f}',
                   xy=(best_layer, best_f1),
                   xytext=(best_layer + 1, best_f1 - 0.05),
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        # Styling
        task_name = result['task'].replace('_', ' ').title()
        model_name = result['model'].split('/')[-1]
        
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{task_name}\n({model_name})', fontsize=13, fontweight='bold')
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=10)
        
        # Add layer labels
        ax.set_xticks(layers[::2])  # Show every other layer
        ax.set_xticklabels([str(l) for l in layers[::2]])
    
    plt.tight_layout()
    output_path = output_dir / 'layer_probing_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()


def plot_comparative_heatmap(results: list, output_dir: Path):
    """Plot heatmap comparing tasks/models across layers.
    
    Args:
        results: List of probing results
        output_dir: Directory for output plots
    """
    if len(results) < 2:
        logger.info("Skipping heatmap (need at least 2 tasks)")
        return
    
    # Prepare data matrix
    task_labels = []
    data_matrix = []
    
    for result in results:
        task_name = result['task'].replace('_', ' ').title()
        task_labels.append(task_name)
        
        f1_scores = [r['f1_score'] for r in result['results']]
        data_matrix.append(f1_scores)
    
    data_matrix = np.array(data_matrix)
    layers = list(range(data_matrix.shape[1]))
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    
    # Set ticks
    ax.set_xticks(layers[::2])
    ax.set_xticklabels([f'L{l}' for l in layers[::2]])
    ax.set_yticks(range(len(task_labels)))
    ax.set_yticklabels(task_labels)
    
    # Labels
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Probing Performance (F1 Score)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Score', fontsize=11)
    
    # Add text annotations
    for i in range(len(task_labels)):
        for j in range(len(layers)):
            if j % 2 == 0:  # Annotate every other layer
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'layer_probing_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()


def plot_knowledge_encoding(results: list, output_dir: Path):
    """Plot where different knowledge types are encoded.
    
    Args:
        results: List of probing results
        output_dir: Directory for output plots
    """
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3A86FF', '#FF006E', '#8338EC', '#FB5607']
    markers = ['o', 's', '^', 'd']
    
    for i, result in enumerate(results):
        task_name = result['task'].replace('_', ' ').title()
        layers = [r['layer'] for r in result['results']]
        f1_scores = [r['f1_score'] for r in result['results']]
        
        # Normalize to show relative performance
        f1_array = np.array(f1_scores)
        normalized = (f1_array - f1_array.min()) / (f1_array.max() - f1_array.min() + 1e-10)
        
        ax.plot(layers, normalized, marker=markers[i % len(markers)],
               linewidth=2.5, markersize=8, label=task_name,
               color=colors[i % len(colors)])
    
    ax.set_xlabel('Layer Depth', fontsize=13, fontweight='bold')
    ax.set_ylabel('Normalized Probe Performance', fontsize=13, fontweight='bold')
    ax.set_title('Knowledge Encoding Across Layers', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)
    
    # Add shaded regions
    num_layers = len(layers)
    ax.axvspan(0, num_layers*0.33, alpha=0.1, color='blue', label='Early layers')
    ax.axvspan(num_layers*0.33, num_layers*0.67, alpha=0.1, color='green')
    ax.axvspan(num_layers*0.67, num_layers, alpha=0.1, color='orange')
    
    # Add annotations
    ax.text(num_layers*0.15, 0.95, 'Surface\nFeatures', 
           ha='center', fontsize=10, style='italic', color='blue', alpha=0.7)
    ax.text(num_layers*0.5, 0.95, 'Syntactic\nPatterns',
           ha='center', fontsize=10, style='italic', color='green', alpha=0.7)
    ax.text(num_layers*0.85, 0.95, 'Semantic\nKnowledge',
           ha='center', fontsize=10, style='italic', color='orange', alpha=0.7)
    
    plt.tight_layout()
    output_path = output_dir / 'knowledge_encoding.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_path}")
    plt.close()


def main():
    if not PLOTTING_AVAILABLE:
        logger.error("matplotlib not available. Install with: pip install matplotlib numpy")
        return
    
    parser = argparse.ArgumentParser(description='Generate probing visualizations')
    parser.add_argument(
        '--results',
        type=Path,
        default=Path('results/probing_results.json'),
        help='Path to probing results JSON'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/plots'),
        help='Directory for output plots'
    )
    
    args = parser.parse_args()
    
    # Load results
    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        return
    
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} probing experiments")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_layer_curves(results, args.output_dir)
    plot_comparative_heatmap(results, args.output_dir)
    plot_knowledge_encoding(results, args.output_dir)
    
    logger.info(f"\nAll plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
