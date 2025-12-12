#!/usr/bin/env python3
"""
Script to normalize Sanskrit text data (Bhagavad Gita).

This script:
1. Loads raw Sanskrit text
2. Applies Unicode normalization
3. Cleans punctuation variants
4. Preserves verse IDs
5. Converts between scripts (Devanagari/IAST)
6. Outputs to data/processed/gita_clean.jsonl

Usage:
    python scripts/normalize_data.py
    python scripts/normalize_data.py --script iast
    python scripts/normalize_data.py --input data/raw/gita.txt --output data/processed/gita_clean.jsonl
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.data.normalizer import DataNormalizer
from sanskriteval.utils.config import load_config
from sanskriteval.utils.logging import setup_logger


def main():
    """Main normalization pipeline."""
    parser = argparse.ArgumentParser(
        description="Normalize Sanskrit text data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/gita_raw.txt",
        help="Input file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/gita_clean.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--script",
        type=str,
        choices=["devanagari", "iast", "slp1"],
        default="iast",
        help="Target script for output (default: iast)"
    )
    parser.add_argument(
        "--input-script",
        type=str,
        choices=["devanagari", "iast", "slp1"],
        default="devanagari",
        help="Script of input text (default: devanagari)"
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["txt", "json", "jsonl"],
        default="txt",
        help="Input file format (default: txt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="bhagavad_gita",
        help="Source text name"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger("normalize_data")
    logger.info("=" * 60)
    logger.info("Sanskrit Data Normalization Pipeline")
    logger.info("=" * 60)
    
    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info(f"Please create a sample file at {input_path}")
        logger.info("See data/raw/README.md for format examples")
        sys.exit(1)
    
    # Initialize normalizer
    logger.info(f"Output script: {args.script}")
    logger.info(f"Input script: {args.input_script}")
    logger.info(f"Source: {args.source}")
    
    normalizer = DataNormalizer(
        output_script=args.script,
        source_name=args.source
    )
    
    # Process file
    try:
        num_verses = normalizer.normalize_file(
            input_path=input_path,
            output_path=output_path,
            input_format=args.input_format,
            input_script=args.input_script
        )
        
        logger.info("=" * 60)
        logger.info(f"✓ Successfully normalized {num_verses} verses")
        logger.info(f"✓ Output saved to: {output_path}")
        logger.info("=" * 60)
        
        # Show sample output
        logger.info("\nSample output (first verse):")
        import jsonlines
        with jsonlines.open(output_path) as reader:
            first = next(iter(reader), None)
            if first:
                logger.info(f"  ID: {first['id']}")
                logger.info(f"  Text: {first['text'][:80]}...")
                logger.info(f"  Script: {first['script']}")
                logger.info(f"  Source: {first['source']}")
        
    except Exception as e:
        logger.error(f"✗ Normalization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
