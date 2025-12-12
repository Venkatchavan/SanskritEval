"""Run layer-wise probing experiments."""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.models.probing import (
    LayerWiseProber,
    prepare_sandhi_probe_data,
    prepare_morphology_probe_data,
    DEPENDENCIES_AVAILABLE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("probe_layers")


def load_sandhi_data(path: Path) -> List[Tuple[str, str, bool]]:
    """Load sandhi data for probing.
    
    Args:
        path: Path to sandhi JSONL file
        
    Returns:
        List of (fused, segmented, has_boundaries) tuples
    """
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            # Check if segmented has boundaries (spaces)
            has_boundaries = ' ' in ex['segmented']
            examples.append((ex['fused'], ex['segmented'], has_boundaries))
    return examples


def load_morphology_data(path: Path) -> List[Tuple[str, str]]:
    """Load morphology contrast pairs.
    
    Args:
        path: Path to contrast set JSONL file
        
    Returns:
        List of (grammatical, ungrammatical) tuples
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            pairs.append((ex['grammatical'], ex['ungrammatical']))
    return pairs


def probe_sandhi(
    model_name: str,
    data_path: Path,
    device: str,
    sample: int = None
) -> dict:
    """Run sandhi boundary probing.
    
    Args:
        model_name: Model to probe
        data_path: Path to sandhi data
        device: Device to use
        sample: Limit number of examples
        
    Returns:
        Results dictionary
    """
    logger.info("="*60)
    logger.info("PROBING: SANDHI BOUNDARIES")
    logger.info("="*60)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    examples = load_sandhi_data(data_path)
    
    if sample:
        examples = examples[:sample]
        logger.info(f"Using sample of {sample} examples")
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # Prepare train/test split
    train_texts, train_labels, test_texts, test_labels = \
        prepare_sandhi_probe_data(examples, train_ratio=0.7)
    
    logger.info(f"Train: {len(train_texts)} examples")
    logger.info(f"Test: {len(test_texts)} examples")
    logger.info(f"Label distribution - Train: {train_labels.sum()}/{len(train_labels)} positive")
    logger.info(f"Label distribution - Test: {test_labels.sum()}/{len(test_labels)} positive")
    
    # Initialize prober
    prober = LayerWiseProber(model_name, device=device)
    
    # Probe all layers
    results = prober.probe_all_layers(
        train_texts, train_labels,
        test_texts, test_labels
    )
    
    # Format results
    return {
        'task': 'sandhi_boundaries',
        'model': model_name,
        'num_layers': prober.num_layers,
        'hidden_dim': prober.hidden_dim,
        'results': [
            {
                'layer': r.layer_idx,
                'accuracy': r.accuracy,
                'f1_score': r.f1_score
            }
            for r in results
        ]
    }


def probe_morphology(
    model_name: str,
    data_path: Path,
    device: str,
    sample: int = None
) -> dict:
    """Run morphology acceptability probing.
    
    Args:
        model_name: Model to probe
        data_path: Path to contrast set data
        device: Device to use
        sample: Limit number of pairs
        
    Returns:
        Results dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("PROBING: MORPHOLOGICAL ACCEPTABILITY")
    logger.info("="*60)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    pairs = load_morphology_data(data_path)
    
    if sample:
        pairs = pairs[:sample]
        logger.info(f"Using sample of {sample} pairs")
    
    logger.info(f"Loaded {len(pairs)} contrast pairs")
    
    # Prepare train/test split
    train_texts, train_labels, test_texts, test_labels = \
        prepare_morphology_probe_data(pairs, train_ratio=0.7)
    
    logger.info(f"Train: {len(train_texts)} examples ({len(train_texts)//2} pairs)")
    logger.info(f"Test: {len(test_texts)} examples ({len(test_texts)//2} pairs)")
    
    # Initialize prober
    prober = LayerWiseProber(model_name, device=device)
    
    # Probe all layers
    results = prober.probe_all_layers(
        train_texts, train_labels,
        test_texts, test_labels
    )
    
    # Format results
    return {
        'task': 'morphological_acceptability',
        'model': model_name,
        'num_layers': prober.num_layers,
        'hidden_dim': prober.hidden_dim,
        'results': [
            {
                'layer': r.layer_idx,
                'accuracy': r.accuracy,
                'f1_score': r.f1_score
            }
            for r in results
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Layer-wise probing experiments')
    parser.add_argument(
        '--task',
        choices=['sandhi', 'morphology', 'both'],
        default='both',
        help='Task to probe'
    )
    parser.add_argument(
        '--model',
        default='bert-base-multilingual-cased',
        help='Model to probe (HuggingFace identifier)'
    )
    parser.add_argument(
        '--sandhi-data',
        type=Path,
        default=Path('data/benchmarks/sandhi_gold_test.jsonl'),
        help='Path to sandhi data'
    )
    parser.add_argument(
        '--morph-data',
        type=Path,
        default=Path('data/benchmarks/morph_contrast_sets.jsonl'),
        help='Path to morphology data'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N examples for quick testing'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/probing_results.json'),
        help='Output JSON file'
    )
    
    args = parser.parse_args()
    
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Missing dependencies. Install: pip install torch transformers scikit-learn")
        return
    
    results = []
    
    # Run sandhi probing
    if args.task in ['sandhi', 'both']:
        try:
            result = probe_sandhi(args.model, args.sandhi_data, args.device, args.sample)
            results.append(result)
        except Exception as e:
            logger.error(f"Sandhi probing failed: {e}")
    
    # Run morphology probing
    if args.task in ['morphology', 'both']:
        try:
            result = probe_morphology(args.model, args.morph_data, args.device, args.sample)
            results.append(result)
        except Exception as e:
            logger.error(f"Morphology probing failed: {e}")
    
    # Save results
    if results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Results saved to {args.output}")
        logger.info(f"{'='*60}")
        
        # Print summary
        for result in results:
            print(f"\n{result['task'].upper()} ({result['model']})")
            print(f"{'Layer':<8} {'Accuracy':<12} {'F1 Score':<12}")
            print("-" * 32)
            
            for r in result['results']:
                print(f"{r['layer']:<8} {r['accuracy']:<12.3f} {r['f1_score']:<12.3f}")
            
            # Find best layer
            best = max(result['results'], key=lambda x: x['f1_score'])
            print(f"\nBest layer: {best['layer']} (F1={best['f1_score']:.3f})")


if __name__ == '__main__':
    main()
