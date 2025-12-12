#!/usr/bin/env python3
"""
Convert Bhagwad_Gita.csv to formats suitable for sandhi dataset generation.

The CSV has columns: ID, Chapter, Verse, Shloka, Transliteration, HinMeaning, EngMeaning, WordMeaning
We'll extract the Shloka (Sanskrit text) column.

Usage:
    python scripts/convert_gita_csv.py
    python scripts/convert_gita_csv.py --output data/raw/gita_full_700.txt
"""

import argparse
import csv
import re
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.utils.logging import setup_logger


def clean_shloka(text: str) -> str:
    """Clean shloka text by removing line breaks and verse numbers.
    
    Args:
        text: Raw shloka text
        
    Returns:
        Cleaned single-line text
    """
    # Remove verse number markers like ||१-१||
    text = re.sub(r'\|\|.*?\|\|', '', text)
    
    # Remove line breaks and extra spaces
    text = text.replace('\n', ' ')
    text = text.replace('|', '')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def convert_csv_to_txt(
    csv_path: Path,
    output_path: Path
) -> int:
    """Convert CSV to simple text format for sandhi generation.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output text file
        
    Returns:
        Number of verses processed
    """
    logger = setup_logger("convert_gita")
    logger.info(f"Converting {csv_path} to {output_path}")
    
    verses = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            verse_id = row.get('ID', '').replace('BG', '')  # BG1.1 -> 1.1
            shloka = row.get('Shloka', '')
            
            if shloka:
                cleaned = clean_shloka(shloka)
                if cleaned:
                    verses.append((verse_id, cleaned))
    
    # Write to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for verse_id, text in verses:
            f.write(f"{verse_id}: {text}\n")
    
    logger.info(f"Converted {len(verses)} verses")
    return len(verses)


def show_statistics(csv_path: Path):
    """Show statistics about the CSV file.
    
    Args:
        csv_path: Path to CSV file
    """
    logger = setup_logger("convert_gita")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Count by chapter
    by_chapter = {}
    for row in rows:
        chapter = row.get('Chapter', '')
        if chapter:
            by_chapter[chapter] = by_chapter.get(chapter, 0) + 1
    
    logger.info(f"\nStatistics:")
    logger.info(f"  Total verses: {len(rows)}")
    logger.info(f"  Chapters: {len(by_chapter)}")
    logger.info(f"\nVerses per chapter:")
    for chapter in sorted(by_chapter.keys(), key=lambda x: int(x)):
        logger.info(f"    Chapter {chapter}: {by_chapter[chapter]} verses")
    
    # Show sample
    if rows:
        logger.info(f"\nSample verse:")
        sample = rows[0]
        logger.info(f"  ID: {sample.get('ID')}")
        logger.info(f"  Shloka: {clean_shloka(sample.get('Shloka', ''))[:80]}...")


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert Bhagwad_Gita.csv to text format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Bhagwad_Gita.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/gita_full_700.txt",
        help="Output text file"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only (no conversion)"
    )
    
    args = parser.parse_args()
    
    logger = setup_logger("convert_gita")
    logger.info("=" * 60)
    logger.info("Bhagavad Gita CSV Converter")
    logger.info("=" * 60)
    
    csv_path = Path(args.input)
    output_path = Path(args.output)
    
    if not csv_path.exists():
        logger.error(f"Input file not found: {csv_path}")
        sys.exit(1)
    
    # Show statistics
    show_statistics(csv_path)
    
    if not args.stats:
        # Convert
        logger.info("\nConverting to text format...")
        num_verses = convert_csv_to_txt(csv_path, output_path)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"✓ Conversion complete!")
        logger.info(f"✓ Output: {output_path}")
        logger.info(f"✓ Verses: {num_verses}")
        logger.info("=" * 60)
        
        logger.info("\nNext steps:")
        logger.info(f"  1. Review: {output_path}")
        logger.info(f"  2. Generate sandhi dataset:")
        logger.info(f"     python scripts/generate_sandhi_data.py \\")
        logger.info(f"       --input {output_path} \\")
        logger.info(f"       --gold-size 200 \\")
        logger.info(f"       --silver-size 700")


if __name__ == "__main__":
    main()
