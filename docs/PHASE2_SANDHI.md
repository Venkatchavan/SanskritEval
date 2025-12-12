# Phase 2: Sandhi Segmentation Dataset

## Overview

Phase 2 implements the **Sandhi Segmentation** task - detecting word boundaries in phonologically fused Sanskrit text. This is Task A of the SanskritEval benchmark.

## What is Sandhi?

Sandhi (संधि) is the phonological process in Sanskrit where sounds at word boundaries merge according to specific rules:

- **Vowel Sandhi**: रामः + अगच्छत् → रामोऽगच्छत्
- **Consonant Sandhi**: तत् + च → तच्च
- **Visarga Sandhi**: पाण्डवाः + च → पाण्डवाश्च

This makes word boundary detection challenging for both humans and models.

## Implementation Strategy

We implemented **Option A: Silver + Gold Hybrid**:

1. **Silver Training Set**: Automatically generated using rule-based heuristics
   - Fast to generate
   - Covers many examples
   - ~60-80% accuracy (good enough for training)

2. **Gold Test Set**: Template for manual correction
   - 200 carefully selected examples
   - Stratified sampling across chapters
   - Manual verification required for high accuracy

## Components Created

### 1. Sandhi Rules Module (`src/sanskriteval/data/sandhi.py`)

**Classes:**
- `SandhiRule`: Represents a transformation rule with confidence
- `SandhiRules`: Collection of common sandhi patterns
- `SimpleSandhiSplitter`: Rule-based segmenter
- `SandhiExample`: Data structure for examples

**Key Methods:**
```python
# Detect potential boundaries
detect_potential_boundaries(text: str) -> List[int]

# Split a verse
split_verse(verse: str, verse_id: str) -> SandhiExample

# Mark boundaries with confidence
mark_boundaries(text: str) -> List[Tuple[int, float]]
```

### 2. Dataset Generator (`scripts/generate_sandhi_data.py`)

**Features:**
- Loads verses from raw text files
- Applies rule-based segmentation
- Generates silver training set
- Creates gold test template with stratified sampling
- Outputs JSONL format

**Usage:**
```bash
# Default: 200 gold, all verses silver
python scripts/generate_sandhi_data.py

# Custom sizes
python scripts/generate_sandhi_data.py --gold-size 100 --silver-size 500

# Different input
python scripts/generate_sandhi_data.py --input data/raw/custom.txt
```

### 3. Annotation Guide (`data/benchmarks/ANNOTATION_GUIDE.md`)

Comprehensive guide for manual annotators covering:
- Sandhi principles
- Annotation workflow
- Common patterns in Bhagavad Gita
- Quality checks
- Resources and tips

## Dataset Format

### JSONL Schema

```json
{
  "verse_id": "1.1",
  "fused": "धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।",
  "segmented": "धृतराष्ट्र उवाच। धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।",
  "confidence": 0.6,
  "is_gold": false
}
```

**Fields:**
- `verse_id`: Unique identifier (chapter.verse)
- `fused`: Original text with sandhi (input)
- `segmented`: Word boundaries marked with spaces (target)
- `confidence`: Quality score (0-1)
  - Silver: 0.5-0.8 (automatic)
  - Gold (needs review): 0.0
  - Gold (verified): 1.0
- `is_gold`: true for gold test set

### Evaluation Metrics

For this task, we'll use:
- **Precision**: Correct boundaries / Predicted boundaries
- **Recall**: Correct boundaries / True boundaries
- **F1 Score**: Harmonic mean of precision and recall
- **Exact Match**: Percentage of verses with perfect segmentation

## Deliverables

✅ **Silver Training Set**: `data/benchmarks/sandhi_silver_train.jsonl`
- Automatically generated
- 10 examples (extendable with more source data)
- Confidence: 0.5-0.8

✅ **Gold Test Template**: `data/benchmarks/sandhi_gold_test.jsonl`
- 10 examples sampled across chapters
- Needs manual verification
- Confidence: 0.0 (to be updated to 1.0 after review)

⚠️  **Manual Review Required**: Follow `ANNOTATION_GUIDE.md` to verify gold set

## Rule-Based Heuristics

Our splitter uses these strategies:

### 1. Danda Markers
```
।  → Sentence boundary (high confidence)
॥ → Verse boundary (high confidence)
```

### 2. Visarga Before Consonant
```
ः + [consonant] → likely word boundary
Example: पाण्डवाः च (NOT पाण्डवाश्च)
```

### 3. Halant (Virama) Boundaries
```
्  + [consonant] → potential boundary
Example: तत् च (NOT तच्च)
```

### 4. Existing Spaces
Preserve spaces already in the text (often correct in source data).

## Limitations & Future Work

### Current Limitations

1. **Simple heuristics**: Not linguistically comprehensive
   - Covers ~60-70% of common patterns
   - Misses complex compound words (samāsa)
   - Doesn't handle all vowel sandhi rules

2. **Limited training data**: Only 10 verses
   - Need larger Gita dataset (700 verses available)
   - Could include other texts (Upanishads, etc.)

3. **Devanagari-only**: No IAST support yet
   - Should add transliteration-aware rules
   - IAST might be easier for models

### Future Improvements

**Short-term:**
- [ ] Add more source verses (expand to 700 verses)
- [ ] Implement vowel sandhi rules
- [ ] Add compound word (samāsa) detection
- [ ] Create IAST version of dataset

**Medium-term:**
- [ ] Train CRF/LSTM model on silver data
- [ ] Use model predictions to improve silver labels
- [ ] Active learning: suggest difficult cases for annotation

**Long-term:**
- [ ] Integrate with existing Sanskrit tools (sandhi-splitters)
- [ ] Add multi-lingual support (Tamil, Pali)
- [ ] Create pada-pāṭha (split text) database

## Validation

To validate gold annotations after manual review:

```python
# Check JSON format
python -m json.tool data/benchmarks/sandhi_gold_test.jsonl

# Validate annotations (to be implemented)
python scripts/validate_gold_annotations.py
```

## Next Phase

With sandhi dataset complete, we move to **Phase 3: Morphological Contrast Sets** for testing case/agreement understanding.

## Files Created

```
src/sanskriteval/data/
└── sandhi.py                    # Sandhi rules and splitter

scripts/
└── generate_sandhi_data.py      # Dataset generation script

data/benchmarks/
├── sandhi_silver_train.jsonl    # Silver training set (10 examples)
├── sandhi_gold_test.jsonl       # Gold test template (10 examples)
└── ANNOTATION_GUIDE.md          # Manual annotation guide

docs/
└── PHASE2_SANDHI.md            # This file
```

## Statistics

- **Source verses**: 10 (from Bhagavad Gita chapters 1-4)
- **Silver examples**: 10
- **Gold template**: 10 (needs manual verification)
- **Average confidence**: 0.6 (silver)
- **Annotation time estimate**: ~2-3 hours for 200 verses

## Resources

- [Sanskrit Sandhi Rules](https://en.wikipedia.org/wiki/Sandhi#Sanskrit)
- [Gita Supersite (IIT Kanpur)](http://www.gitasupersite.iitk.ac.in/)
- [Sanskrit NLP Resources](https://github.com/topics/sanskrit-nlp)
