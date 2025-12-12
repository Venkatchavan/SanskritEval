#!/usr/bin/env python3
"""
Generate morphological contrast sets (minimal pairs) for evaluating
language model understanding of Sanskrit case and number inflection.

Usage:
    python scripts/generate_morph_data.py
    python scripts/generate_morph_data.py --target-size 500 --max-stems 200
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.data.contrast_sets import (
    ContrastSetGenerator,
    NounExtractor
)
from sanskriteval.data.morphology import SanskritDeclensions
from sanskriteval.utils.logging import setup_logger


def main():
    """Main generation pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate morphological contrast sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gita_full_700.txt",
        help="Input corpus file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmarks/morph_contrast_sets.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=500,
        help="Target number of contrast pairs"
    )
    parser.add_argument(
        "--max-stems",
        type=int,
        default=200,
        help="Maximum number of unique stems to extract"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("generate_morph")
    logger.info("=" * 60)
    logger.info("Morphological Contrast Set Generation")
    logger.info("=" * 60)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Step 1: Extract nouns from corpus
    logger.info(f"\n[1/3] Extracting nouns from {input_path}")
    extractor = NounExtractor()
    nouns = extractor.extract_from_corpus(input_path, max_nouns=args.max_stems)
    logger.info(f"  Extracted {len(nouns)} unique noun stems")
    
    # Show statistics
    patterns_count = {}
    for word, stem, pattern in nouns:
        key = f"{pattern.stem_class} ({pattern.gender.value})"
        patterns_count[key] = patterns_count.get(key, 0) + 1
    
    logger.info(f"\n  Breakdown by declension:")
    for pattern_name, count in sorted(patterns_count.items()):
        logger.info(f"    {pattern_name}: {count} stems")
    
    # Show sample nouns
    logger.info(f"\n  Sample extracted nouns:")
    for word, stem, pattern in nouns[:5]:
        logger.info(f"    {word} (stem: {stem}, {pattern.stem_class})")
    
    # Step 2: Generate contrast pairs
    logger.info(f"\n[2/3] Generating {args.target_size} contrast pairs...")
    generator = ContrastSetGenerator(seed=args.seed)
    pairs = generator.generate_dataset(nouns, target_size=args.target_size)
    logger.info(f"  Generated {len(pairs)} pairs")
    
    # Show phenomenon breakdown
    by_phenomenon = {}
    for pair in pairs:
        by_phenomenon[pair.phenomenon] = by_phenomenon.get(pair.phenomenon, 0) + 1
    
    logger.info(f"\n  Breakdown by phenomenon:")
    for phenom, count in sorted(by_phenomenon.items()):
        logger.info(f"    {phenom}: {count} pairs")
    
    # Step 3: Save to file
    logger.info(f"\n[3/3] Saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            json.dump(pair.to_dict(), f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"  Saved {len(pairs)} pairs")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Contrast set generation complete!")
    logger.info("=" * 60)
    logger.info(f"\nOutput: {output_path}")
    logger.info(f"Total pairs: {len(pairs)}")
    logger.info(f"Unique stems: {len(nouns)}")
    
    # Show sample pairs
    logger.info(f"\nSample contrast pairs:")
    for i, pair in enumerate(pairs[:3], 1):
        logger.info(f"\n  Pair {i} (ID: {pair.id}):")
        logger.info(f"    Phenomenon: {pair.phenomenon}")
        logger.info(f"    ✓ Grammatical:   {pair.grammatical}")
        logger.info(f"    ✗ Ungrammatical: {pair.ungrammatical}")
        logger.info(f"    Stem: {pair.stem}")
        if pair.phenomenon == "case":
            logger.info(f"    {pair.metadata['correct_case']} → {pair.metadata['incorrect_case']}")
        else:
            logger.info(f"    {pair.metadata['correct_number']} → {pair.metadata['incorrect_number']}")
    
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review: {output_path}")
    logger.info(f"  2. Evaluate models with this dataset")
    logger.info(f"  3. Calculate acceptability accuracy")


if __name__ == "__main__":
    main()
