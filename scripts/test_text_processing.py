#!/usr/bin/env python3
"""
Quick test of normalization without external dependencies.
Tests core text processing functions.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sanskriteval.utils.text_processing import (
    normalize_unicode,
    remove_zero_width_chars,
    normalize_punctuation,
    clean_sanskrit_text,
    extract_verse_id
)


def test_unicode_normalization():
    """Test Unicode normalization."""
    print("Testing Unicode normalization...")
    text = "कृष्ण"  # Krishna in Devanagari
    normalized = normalize_unicode(text)
    print(f"  Original: {text}")
    print(f"  Normalized: {normalized}")
    print(f"  ✓ Passed\n")


def test_punctuation_normalization():
    """Test punctuation normalization."""
    print("Testing punctuation normalization...")
    text = "धर्मक्षेत्रे  कुरुक्षेत्रे।।"  # Multiple spaces
    cleaned = normalize_punctuation(text)
    print(f"  Original: '{text}'")
    print(f"  Cleaned: '{cleaned}'")
    assert "  " not in cleaned, "Multiple spaces should be normalized"
    print(f"  ✓ Passed\n")


def test_verse_id_extraction():
    """Test verse ID extraction."""
    print("Testing verse ID extraction...")
    test_cases = [
        ("1.1: धर्मक्षेत्रे कुरुक्षेत्रे", "1.1"),
        ("2.15 - सञ्जय उवाच", "2.15"),
        ("3.42: इन्द्रियाणि पराण्याहुः", "3.42"),
    ]
    
    for text, expected in test_cases:
        extracted = extract_verse_id(text)
        print(f"  Text: {text[:30]}...")
        print(f"  Extracted: {extracted}")
        assert extracted == expected, f"Expected {expected}, got {extracted}"
    print(f"  ✓ All {len(test_cases)} cases passed\n")


def test_full_cleaning():
    """Test full text cleaning pipeline."""
    print("Testing full cleaning pipeline...")
    text = "1.1:  धर्मक्षेत्रे   कुरुक्षेत्रे\u200b समवेता युयुत्सवः।।  "
    cleaned = clean_sanskrit_text(text)
    print(f"  Original: '{text}'")
    print(f"  Cleaned: '{cleaned}'")
    assert "\u200b" not in cleaned, "Zero-width space should be removed"
    assert not cleaned.startswith(" "), "Leading spaces should be removed"
    assert not cleaned.endswith(" "), "Trailing spaces should be removed"
    print(f"  ✓ Passed\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Text Processing Unit Tests")
    print("=" * 60)
    print()
    
    try:
        test_unicode_normalization()
        test_punctuation_normalization()
        test_verse_id_extraction()
        test_full_cleaning()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
