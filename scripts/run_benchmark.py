"""Run full benchmark evaluation and generate results."""

import json
import logging
import argparse
from pathlib import Path
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("run_benchmark")


def run_sandhi_eval(test_file: Path, output_file: Path) -> bool:
    """Run sandhi segmentation evaluation.
    
    Args:
        test_file: Path to test data
        output_file: Path for results
        
    Returns:
        True if successful
    """
    logger.info("="*60)
    logger.info("TASK A: SANDHI SEGMENTATION")
    logger.info("="*60)
    
    cmd = [
        sys.executable,
        'scripts/evaluate_sandhi.py',
        '--test-file', str(test_file),
        '--output', str(output_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Sandhi evaluation failed: {e}")
        print(e.stderr)
        return False


def run_morphology_eval(
    test_file: Path,
    output_file: Path,
    models: list,
    device: str = 'cpu',
    sample: int = None
) -> bool:
    """Run morphology acceptability evaluation.
    
    Args:
        test_file: Path to contrast set data
        output_file: Path for results
        models: List of model keys to evaluate
        device: Device to use
        sample: Sample size for quick testing
        
    Returns:
        True if successful
    """
    logger.info("\n" + "="*60)
    logger.info("TASK B: MORPHOLOGICAL ACCEPTABILITY")
    logger.info("="*60)
    
    cmd = [
        sys.executable,
        'scripts/evaluate_morphology.py',
        '--test-file', str(test_file),
        '--output', str(output_file),
        '--models', *models,
        '--device', device
    ]
    
    if sample:
        cmd.extend(['--sample', str(sample)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Morphology evaluation failed: {e}")
        print(e.stderr)
        return False


def combine_results(sandhi_file: Path, morph_file: Path, output_file: Path):
    """Combine results into summary CSV.
    
    Args:
        sandhi_file: Path to sandhi results JSON
        morph_file: Path to morphology results JSON
        output_file: Path for combined CSV
    """
    import csv
    
    logger.info(f"\nGenerating summary CSV: {output_file}")
    
    # Load results
    with open(sandhi_file, 'r') as f:
        sandhi_results = json.load(f)
    
    with open(morph_file, 'r') as f:
        morph_results = json.load(f)
    
    # Prepare CSV data
    rows = []
    
    # Header
    rows.append([
        'Model',
        'Task',
        'Metric',
        'Score',
        'Details'
    ])
    
    # Sandhi results
    for result in sandhi_results:
        model = result['model']
        rows.append([model, 'Sandhi Segmentation', 'Precision', f"{result['precision']:.3f}", ''])
        rows.append([model, 'Sandhi Segmentation', 'Recall', f"{result['recall']:.3f}", ''])
        rows.append([model, 'Sandhi Segmentation', 'F1', f"{result['f1']:.3f}", ''])
        rows.append([model, 'Sandhi Segmentation', 'Exact Match', f"{result['exact_match']:.3f}", ''])
    
    # Morphology results
    for result in morph_results:
        model = result['model']
        rows.append([model, 'Morphological Acceptability', 'Accuracy', f"{result['accuracy']:.3f}", ''])
        
        # Add phenomenon breakdown
        for phenom, acc in result['by_phenomenon'].items():
            rows.append([model, 'Morphological Acceptability', f'Accuracy ({phenom})', f"{acc:.3f}", ''])
    
    # Write CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    logger.info(f"Summary saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run full SanskritEval benchmark')
    parser.add_argument(
        '--sandhi-test',
        type=Path,
        default=Path('data/benchmarks/sandhi_gold_test.jsonl'),
        help='Path to sandhi test data'
    )
    parser.add_argument(
        '--morph-test',
        type=Path,
        default=Path('data/benchmarks/morph_contrast_sets.jsonl'),
        help='Path to morphology contrast sets'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['mbert', 'xlm-r-base'],
        help='Models to evaluate for morphology'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for model evaluation'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Sample N pairs for quick testing (morphology only)'
    )
    parser.add_argument(
        '--skip-sandhi',
        action='store_true',
        help='Skip sandhi evaluation'
    )
    parser.add_argument(
        '--skip-morphology',
        action='store_true',
        help='Skip morphology evaluation'
    )
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run evaluations
    success = True
    
    if not args.skip_sandhi:
        sandhi_output = results_dir / f'sandhi_results_{timestamp}.json'
        if not run_sandhi_eval(args.sandhi_test, sandhi_output):
            success = False
    else:
        sandhi_output = results_dir / 'sandhi_results.json'
    
    if not args.skip_morphology:
        morph_output = results_dir / f'morphology_results_{timestamp}.json'
        if not run_morphology_eval(
            args.morph_test,
            morph_output,
            args.models,
            args.device,
            args.sample
        ):
            success = False
    else:
        morph_output = results_dir / 'morphology_results.json'
    
    # Combine results
    if success:
        summary_output = results_dir / f'summary_{timestamp}.csv'
        combine_results(sandhi_output, morph_output, summary_output)
        
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved to:")
        logger.info(f"  - {sandhi_output}")
        logger.info(f"  - {morph_output}")
        logger.info(f"  - {summary_output}")
    else:
        logger.warning("\nSome evaluations failed. Check logs above.")


if __name__ == '__main__':
    main()
