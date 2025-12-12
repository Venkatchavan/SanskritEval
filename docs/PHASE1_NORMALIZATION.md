# Phase 1: Data Normalization

## Overview
Phase 1 implements robust text normalization for Sanskrit texts, ensuring consistent representation across different Unicode forms and scripts.

## Implementation

### Components Created

1. **Text Processing Utilities** (`src/sanskriteval/utils/text_processing.py`)
   - Unicode normalization (NFC form)
   - Zero-width character removal
   - Punctuation normalization
   - Script conversion (Devanagari ↔ IAST ↔ SLP1)
   - Verse ID extraction

2. **Data Normalizer** (`src/sanskriteval/data/normalizer.py`)
   - `NormalizedVerse` dataclass for structured output
   - `DataNormalizer` class with support for:
     - Multiple input formats (TXT, JSON, JSONL)
     - Script conversion
     - Batch processing
     - Verse metadata extraction (chapter, verse number)

3. **Normalization Script** (`scripts/normalize_data.py`)
   - Command-line interface
   - Configurable input/output paths
   - Script selection (Devanagari/IAST/SLP1)
   - Progress logging

### Design Decisions

**Script Representation:**
- **Default: IAST** (International Alphabet of Sanskrit Transliteration)
  - Easier debugging (ASCII-based)
  - Better tokenization by standard LLMs
  - Lossless conversion from Devanagari
- **Alternative: Devanagari** (native script)
  - Preserves original form
  - Better for human verification
  - Option available via `--script devanagari`

**Unicode Normalization:**
- NFC (Canonical Composition) for consistency
- Removes zero-width joiners/spaces
- Normalizes various punctuation forms (dandas, quotes, spaces)

**Verse ID Format:**
- Pattern: `{chapter}.{verse}` (e.g., "1.1", "2.15")
- Preserved in `id` field
- Parsed into separate `chapter` and `verse` fields

### Output Format

```json
{
  "id": "1.1",
  "text": "dhṛtarāṣṭra uvāca | dharmakṣetre kurukṣetre samavetā yuyutsavaḥ |",
  "source": "bhagavad_gita",
  "script": "iast",
  "chapter": 1,
  "verse": 1
}
```

## Usage

### Basic Usage
```bash
# Normalize Devanagari text to IAST
python scripts/normalize_data.py

# Keep Devanagari script
python scripts/normalize_data.py --script devanagari

# Custom paths
python scripts/normalize_data.py \
    --input data/raw/custom.txt \
    --output data/processed/custom_clean.jsonl
```

### Testing
```bash
# Run unit tests (no dependencies required)
python scripts/test_text_processing.py
```

## Dependencies

**Core (no external deps):**
- Unicode normalization
- Punctuation cleaning
- Verse ID extraction

**Optional (requires installation):**
- `indic-transliteration` - for script conversion
- `jsonlines` - for JSONL I/O

Install with:
```bash
pip install indic-transliteration jsonlines
```

## Sample Data

Included sample data in `data/raw/gita_raw.txt`:
- 10 verses from Bhagavad Gita (Chapters 1-4)
- Devanagari script
- Format: `verse_id: text`

## Next Steps

With normalized data, we can now:
1. **Phase 2**: Generate sandhi segmentation task data
2. **Phase 3**: Create morphological contrast sets
3. Build evaluation pipeline

## Verification

✅ Core text processing works without external dependencies  
✅ Unicode normalization handles various forms  
✅ Punctuation normalized to standard forms  
✅ Verse ID extraction working for multiple patterns  
✅ Zero-width characters removed  
✅ Sample data included for testing  

## Files Created

```
src/sanskriteval/
├── utils/
│   ├── config.py           # Configuration management
│   ├── logging.py          # Logging utilities
│   └── text_processing.py  # Text normalization functions
└── data/
    └── normalizer.py       # Data normalization class

scripts/
├── normalize_data.py       # Main normalization script
└── test_text_processing.py # Unit tests

data/raw/
├── README.md               # Data format documentation
└── gita_raw.txt           # Sample Bhagavad Gita verses
```
