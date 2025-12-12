"""Evaluate morphological acceptability models."""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.models import ModelFactory, TRANSFORMERS_AVAILABLE
from sanskriteval.metrics import compute_morphology_metrics, format_morphology_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("eval_morph")


def load_test_data(path: Path) -> Tuple[List[Tuple[str, str]], List[str], List[str]]:
    """Load contrast set data from JSONL.
    
    Args:
        path: Path to contrast set JSONL file
        
    Returns:
        Tuple of (pairs, phenomena, stem_classes)
        - pairs: List of (grammatical, ungrammatical) tuples
        - phenomena: List of phenomenon labels
        - stem_classes: List of stem class labels
    """
    pairs = []
    phenomena = []
    stem_classes = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            pairs.append((example['grammatical'], example['ungrammatical']))
            phenomena.append(example['phenomenon'])
            stem_classes.append(example['metadata']['stem_class'])
    
    return pairs, phenomena, stem_classes


def evaluate_model(
    model_key: str,
    pairs: List[Tuple[str, str]],
    phenomena: List[str],
    stem_classes: List[str],
    device: str = 'cpu'
) -> dict:
    """Evaluate a model on contrast sets.
    
    Args:
        model_key: Model identifier
        pairs: List of (grammatical, ungrammatical) tuples
        phenomena: List of phenomenon labels
        stem_classes: List of stem class labels
        device: Device to use
        
    Returns:
        Results dictionary
    """
    logger.info(f"Evaluating model: {model_key}")
    
    # Create scorer
    scorer = ModelFactory.create_scorer(model_key, device=device)
    
    # Score all pairs
    scores = []
    for i, (gram, ungram) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            logger.info(f"  Scored {i+1}/{len(pairs)} pairs...")
        
        gram_score = scorer.score(gram)
        ungram_score = scorer.score(ungram)
        scores.append((gram_score, ungram_score))
    
    # Compute metrics
    metrics = compute_morphology_metrics(scores, phenomena, stem_classes)
    
    logger.info(f"\n{format_morphology_metrics(metrics)}")
    
    return {
        'model': model_key,
        'accuracy': metrics.accuracy,
        'total_pairs': metrics.total_pairs,
        'correct_pairs': metrics.correct_pairs,
        'by_phenomenon': metrics.by_phenomenon,
        'by_stem_class': metrics.by_stem_class
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate morphological acceptability models')
    parser.add_argument(
        '--test-file',
        type=Path,
        default=Path('data/benchmarks/morph_contrast_sets.jsonl'),
        help='Path to contrast set JSONL file'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['mbert', 'xlm-r-base'],
        help='Models to evaluate (e.g., mbert xlm-r-base indicbert)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for computation'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/morphology_results.json'),
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N pairs for quick testing'
    )
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("transformers library not available. Install with: pip install torch transformers")
        return
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    pairs, phenomena, stem_classes = load_test_data(args.test_file)
    logger.info(f"Loaded {len(pairs)} contrast pairs")
    
    # Sample if requested
    if args.sample:
        logger.info(f"Sampling {args.sample} pairs for testing")
        pairs = pairs[:args.sample]
        phenomena = phenomena[:args.sample]
        stem_classes = stem_classes[:args.sample]
    
    # Evaluate models
    results = []
    
    for model_key in args.models:
        try:
            result = evaluate_model(model_key, pairs, phenomena, stem_classes, args.device)
            results.append(result)
        except Exception as e:
            logger.error(f"Error evaluating {model_key}: {e}")
            continue
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("MORPHOLOGICAL ACCEPTABILITY RESULTS")
    print("="*60)
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"  Overall Accuracy: {result['accuracy']:.3f} ({result['correct_pairs']}/{result['total_pairs']})")
        print(f"\n  By Phenomenon:")
        for phenom, acc in sorted(result['by_phenomenon'].items()):
            print(f"    {phenom:10s}: {acc:.3f}")


if __name__ == '__main__':
    main()
