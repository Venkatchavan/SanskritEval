"""Evaluate sandhi segmentation models."""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.models import RuleBasedSegmenter
from sanskriteval.metrics import compute_sandhi_metrics, format_sandhi_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("eval_sandhi")


def load_test_data(path: Path) -> List[Tuple[str, str]]:
    """Load test data from JSONL.
    
    Args:
        path: Path to test JSONL file
        
    Returns:
        List of (fused, segmented) tuples
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line)
            data.append((example['fused'], example['segmented']))
    return data


def evaluate_rule_based(test_data: List[Tuple[str, str]]) -> dict:
    """Evaluate rule-based segmenter.
    
    Args:
        test_data: List of (fused, gold_segmented) tuples
        
    Returns:
        Results dictionary
    """
    logger.info("Evaluating rule-based segmenter...")
    
    segmenter = RuleBasedSegmenter()
    
    # Generate predictions
    predictions = []
    for fused, gold_seg in test_data:
        pred_seg = segmenter.segment(fused)
        predictions.append((fused, pred_seg))
    
    # Compute metrics
    metrics = compute_sandhi_metrics(predictions, test_data)
    
    logger.info(f"\n{format_sandhi_metrics(metrics)}")
    
    return {
        'model': 'rule-based',
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1': metrics.f1,
        'exact_match': metrics.exact_match,
        'total_examples': metrics.total_examples
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate sandhi segmentation models')
    parser.add_argument(
        '--test-file',
        type=Path,
        default=Path('data/benchmarks/sandhi_gold_test.jsonl'),
        help='Path to test JSONL file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/sandhi_results.json'),
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_data = load_test_data(args.test_file)
    logger.info(f"Loaded {len(test_data)} examples")
    
    # Evaluate models
    results = []
    
    # Rule-based baseline
    result = evaluate_rule_based(test_data)
    results.append(result)
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n" + "="*60)
    print("SANDHI SEGMENTATION RESULTS")
    print("="*60)
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"  Precision:   {result['precision']:.3f}")
        print(f"  Recall:      {result['recall']:.3f}")
        print(f"  F1 Score:    {result['f1']:.3f}")
        print(f"  Exact Match: {result['exact_match']:.3f}")


if __name__ == '__main__':
    main()
