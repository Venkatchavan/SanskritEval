# Phase 3: Morphological Contrast Sets

## Overview

This phase implements **Task B: Morphological Acceptability** through minimal pair generation. The task evaluates whether language models can distinguish grammatically correct Sanskrit morphological forms from minimal violations.

## Objective

Generate 300-500 contrast pairs that test a model's understanding of:
- **Case inflection** (8 cases: nominative, accusative, instrumental, dative, ablative, genitive, locative, vocative)
- **Number agreement** (3 numbers: singular, dual, plural)
- **Declension patterns** (a-stem masculine/neuter/feminine)

## Methodology

### 1. Noun Extraction (`NounExtractor`)

**Input**: 701 verses from Bhagavad Gita  
**Process**:
- Pattern match common declension endings (e.g., `ः`, `म्`, `आः`, etc.)
- Extract noun stems by removing recognized endings
- Classify by declension pattern (currently: a-stem masculine)
- Deduplicate and limit to `--max-stems` most frequent

**Output**: List of `(stem, declension_pattern)` tuples

### 2. Contrast Pair Generation (`ContrastSetGenerator`)

**Two perturbation types**:

#### Case Perturbations
- **Keep**: stem, number constant
- **Vary**: case ending
- **Example**: 
  - Grammatical: `राजन् + अस्य` (genitive singular) → `राजअस्य`
  - Ungrammatical: `राजन् + ए` (locative singular) → `राजए`
  - Minimal difference: genitive vs. locative

#### Number Perturbations
- **Keep**: stem, case constant
- **Vary**: number ending
- **Example**:
  - Grammatical: `राजन् + आः` (nominative plural) → `राजआः`
  - Ungrammatical: `राजन् + औ` (nominative dual) → `राजऔ`
  - Minimal difference: plural vs. dual

### 3. Data Format

```json
{
  "id": "morph_0001",
  "phenomenon": "case",
  "grammatical": "राजअस्य",
  "ungrammatical": "राजए",
  "stem": "राज",
  "context": "",
  "metadata": {
    "stem_class": "a-stem",
    "gender": "masculine",
    "number": "singular",
    "correct_case": "genitive",
    "incorrect_case": "locative",
    "ending_from": "अस्य",
    "ending_to": "ए"
  }
}
```

## Generation Pipeline

### Usage

```bash
# Default: 500 pairs from 200 stems
python scripts/generate_morph_data.py

# Custom size
python scripts/generate_morph_data.py --target-size 300 --max-stems 100

# Set random seed for reproducibility
python scripts/generate_morph_data.py --seed 42
```

### Output

**File**: `data/benchmarks/morph_contrast_sets.jsonl`  
**Size**: 500 pairs (333 case + 167 number)  
**Stems**: 200 unique noun stems from corpus

### Statistics (500-pair dataset)

- **Case perturbations**: ~67% (333 pairs)
  - All 8×8 case combinations (excluding identity)
  - 56 possible case swaps per stem
- **Number perturbations**: ~33% (167 pairs)
  - All 3×3 number combinations (excluding identity)
  - 6 possible number swaps per stem
- **Unique stems**: 200 (extracted from 701 verses)

## Evaluation Protocol

### Metric: Acceptability Accuracy

```
accuracy = (# pairs where P(grammatical) > P(ungrammatical)) / total_pairs
```

### Measurement Methods

1. **Perplexity-based** (for causal LMs):
   ```python
   score_gram = model.score("grammatical_form")
   score_ungram = model.score("ungrammatical_form")
   correct = score_gram < score_ungram  # Lower perplexity = better
   ```

2. **Pseudo-likelihood** (for masked LMs):
   ```python
   score_gram = model.score_masked("grammatical_form")
   score_ungram = model.score_masked("ungrammatical_form")
   correct = score_gram > score_ungram  # Higher likelihood = better
   ```

### Expected Baselines

- **Random**: 50%
- **Frequency-based**: 55-60%
- **Transformer-based**: 70-85% (varies by pretraining corpus)

## Declension Patterns Implemented

### Masculine a-stem (e.g., राज "king")

| Case | Singular | Dual | Plural |
|------|----------|------|--------|
| Nominative | ः | औ | आः |
| Accusative | म् | औ | आन् |
| Instrumental | एन | आभ्याम् | ऐः |
| Dative | आय | आभ्याम् | एभ्यः |
| Ablative | आत् | आभ्याम् | एभ्यः |
| Genitive | अस्य | अयोः | आनाम् |
| Locative | ए | अयोः | एषु |
| Vocative | (stem) | औ | आः |

### Neuter a-stem (e.g., फल "fruit")

| Case | Singular | Dual | Plural |
|------|----------|------|--------|
| Nominative | म् | ए | आनि |
| Accusative | म् | ए | आनि |
| (Other cases same as masculine) | | | |

### Feminine ā-stem (e.g., कथा "story")

| Case | Singular | Dual | Plural |
|------|----------|------|--------|
| Nominative | आ | ए | आः |
| Accusative | आम् | ए | आः |
| Instrumental | अया | आभ्याम् | आभिः |
| Dative | आयै | आभ्याम् | आभ्यः |
| (Other cases follow similar patterns) | | | |

## Future Extensions

### Additional Declensions
- i-stems (masculine/feminine/neuter)
- u-stems (masculine/feminine/neuter)
- Consonant stems (n-stems, s-stems, etc.)

### Additional Phenomena
- **Gender agreement**: Adjective-noun mismatches
- **Verb conjugation**: Tense/aspect/mood errors
- **Sandhi violations**: Morpheme boundary errors
- **Compound formation**: Invalid samāsa patterns

### Corpus Expansion
- Add verses from other texts (Rāmāyaṇa, Mahābhārata)
- Include prose passages for more diverse morphology
- Balance distribution across cases/numbers

## Implementation Files

- `src/sanskriteval/data/morphology.py`: Declension pattern definitions
- `src/sanskriteval/data/contrast_sets.py`: Extraction and generation logic
- `scripts/generate_morph_data.py`: CLI interface
- `data/benchmarks/morph_contrast_sets.jsonl`: Output dataset

## References

- Whitney, W. D. (1889). *Sanskrit Grammar*. Harvard University Press.
- Kale, M. R. (1961). *A Higher Sanskrit Grammar*. Motilal Banarsidass.
- Marelli, M., & Baroni, M. (2015). "Affixation in semantic space". *SemEval*.

## Next Steps

1. ✓ Generate morphological contrast sets (500 pairs)
2. ⚠️ Implement evaluation pipeline
3. ⚠️ Evaluate models (GPT-2, mBERT, IndicBERT, etc.)
4. ⚠️ Analyze error patterns by case/number/stem
5. ⚠️ Generate results tables and plots
