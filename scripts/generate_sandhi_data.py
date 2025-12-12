#!/usr/bin/env python3
"""
Generate sandhi segmentation benchmark dataset.

Creates two datasets:
1. Silver training set: Automatically segmented using heuristics
2. Gold test set: Template for manual correction (200 examples)

Usage:
    python scripts/generate_sandhi_data.py
    python scripts/generate_sandhi_data.py --gold-size 200 --silver-size 500
"""

import argparse
import json
from pathlib import Path
import sys
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.data.sandhi import SimpleSandhiSplitter, SandhiExample, create_manual_gold_template
from sanskriteval.utils.logging import setup_logger


def load_verses_from_file(file_path: Path) -> list:
    """Load verses from raw text file.
    
    Args:
        file_path: Path to input file
        
    Returns:
        List of (verse_id, text) tuples
    """
    verses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to parse verse_id: text format
            if ':' in line:
                parts = line.split(':', 1)
                verse_id = parts[0].strip()
                text = parts[1].strip()
                verses.append((verse_id, text))
    
    return verses


def generate_silver_dataset(
    input_path: Path,
    output_path: Path,
    max_samples: int = None
) -> list:
    """Generate silver label dataset using automatic segmentation.
    
    Args:
        input_path: Path to input verses
        output_path: Path to output JSONL
        max_samples: Maximum number of samples (None = all)
        
    Returns:
        List of SandhiExample objects
    """
    logger = setup_logger("generate_sandhi")
    logger.info(f"Loading verses from {input_path}")
    
    verses = load_verses_from_file(input_path)
    logger.info(f"Loaded {len(verses)} verses")
    
    if max_samples:
        verses = verses[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Initialize splitter
    splitter = SimpleSandhiSplitter(confidence_threshold=0.5)
    
    # Process verses
    examples = []
    logger.info("Generating silver segmentations...")
    
    for verse_id, text in verses:
        try:
            example = splitter.split_verse(text, verse_id)
            examples.append(example)
        except Exception as e:
            logger.warning(f"Failed to process verse {verse_id}: {e}")
    
    # Save to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            json.dump(example.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Saved {len(examples)} examples to {output_path}")
    
    return examples


def generate_gold_template(
    silver_examples: list,
    output_path: Path,
    num_samples: int = 200,
    seed: int = 42
) -> None:
    """Generate gold test set template for manual annotation.
    
    Args:
        silver_examples: List of silver examples
        output_path: Path to output JSONL
        num_samples: Number of samples for gold set
        seed: Random seed for sampling
    """
    logger = setup_logger("generate_sandhi")
    logger.info(f"Creating gold template with {num_samples} samples")
    
    # Sample examples (stratified by chapter if possible)
    random.seed(seed)
    
    # Group by chapter
    by_chapter = {}
    for ex in silver_examples:
        chapter = ex.verse_id.split('.')[0] if '.' in ex.verse_id else 'unknown'
        if chapter not in by_chapter:
            by_chapter[chapter] = []
        by_chapter[chapter].append(ex)
    
    # Sample proportionally from each chapter
    gold_examples = []
    chapters = sorted(by_chapter.keys())
    samples_per_chapter = max(1, num_samples // len(chapters))
    
    for chapter in chapters:
        chapter_examples = by_chapter[chapter]
        n_samples = min(samples_per_chapter, len(chapter_examples))
        sampled = random.sample(chapter_examples, n_samples)
        gold_examples.extend(sampled)
    
    # If we still need more, sample from remaining
    if len(gold_examples) < num_samples:
        remaining = [ex for ex in silver_examples if ex not in gold_examples]
        additional = random.sample(
            remaining,
            min(num_samples - len(gold_examples), len(remaining))
        )
        gold_examples.extend(additional)
    
    # Truncate to exact number
    gold_examples = gold_examples[:num_samples]
    
    # Mark for manual review
    for example in gold_examples:
        example.confidence = 0.0  # Needs verification
        example.is_gold = True  # Will be gold after manual correction
    
    # Save to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in gold_examples:
            json.dump(example.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Saved {len(gold_examples)} gold templates to {output_path}")
    logger.info("⚠️  MANUAL REVIEW REQUIRED:")
    logger.info("   1. Open the gold template file")
    logger.info("   2. Verify and correct each 'segmented' field")
    logger.info("   3. Update 'confidence' to 1.0 after verification")
    logger.info("   4. Ensure 'is_gold' is true")


def main():
    """Main pipeline for sandhi dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate sandhi segmentation benchmark dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gita_raw.txt",
        help="Input file with Sanskrit verses"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmarks",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--gold-size",
        type=int,
        default=200,
        help="Number of examples for gold test set"
    )
    parser.add_argument(
        "--silver-size",
        type=int,
        default=None,
        help="Maximum examples for silver training set (None = all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("generate_sandhi")
    logger.info("=" * 60)
    logger.info("Sandhi Segmentation Dataset Generation")
    logger.info("=" * 60)
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Generate silver dataset
    logger.info("\n[1/2] Generating silver training set...")
    silver_path = output_dir / "sandhi_silver_train.jsonl"
    silver_examples = generate_silver_dataset(
        input_path,
        silver_path,
        max_samples=args.silver_size
    )
    
    # Generate gold template
    logger.info("\n[2/2] Generating gold test template...")
    gold_path = output_dir / "sandhi_gold_test.jsonl"
    generate_gold_template(
        silver_examples,
        gold_path,
        num_samples=min(args.gold_size, len(silver_examples)),
        seed=args.seed
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Dataset generation complete!")
    logger.info("=" * 60)
    logger.info(f"\nSilver set: {silver_path}")
    logger.info(f"Gold template: {gold_path}")
    logger.info(f"\n⚠️  Remember to manually review the gold test set!")
    
    # Show statistics
    logger.info(f"\nStatistics:")
    logger.info(f"  Total verses processed: {len(silver_examples)}")
    logger.info(f"  Silver training examples: {len(silver_examples)}")
    logger.info(f"  Gold test examples (needs review): {args.gold_size}")
    
    # Show sample
    if silver_examples:
        logger.info(f"\nSample segmentation:")
        example = silver_examples[0]
        logger.info(f"  Verse ID: {example.verse_id}")
        logger.info(f"  Fused:     {example.fused[:80]}...")
        logger.info(f"  Segmented: {example.segmented[:80]}...")
        logger.info(f"  Confidence: {example.confidence}")


if __name__ == "__main__":
    main()
