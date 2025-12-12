# Phase 4: Model Evaluation & Baselines

## Overview

This phase implements comprehensive evaluation infrastructure for both SanskritEval tasks:
- **Task A**: Sandhi Segmentation
- **Task B**: Morphological Acceptability

## Evaluation Framework

### Architecture

```
scripts/
├── evaluate_sandhi.py         # Sandhi segmentation evaluation
├── evaluate_morphology.py     # Morphology acceptability evaluation
├── run_benchmark.py            # Unified benchmark runner
└── generate_plots.py           # Results visualization

src/sanskriteval/
├── metrics/
│   ├── sandhi_metrics.py       # P/R/F1 for boundary detection
│   └── morphology_metrics.py   # Acceptability accuracy
└── models/
    ├── base.py                 # Abstract interfaces
    ├── rule_based.py           # Rule-based baseline
    └── transformers_models.py  # Transformer scorers
```

## Task A: Sandhi Segmentation

### Objective
Segment Sanskrit text at morpheme boundaries (sandhi splitting).

### Metrics

**Boundary-level evaluation**:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of P & R
- **Exact Match**: % of verses with perfect segmentation

**Boundary detection**:
- True Positive (TP): Predicted boundary matches gold boundary
- False Positive (FP): Predicted boundary not in gold
- False Negative (FN): Gold boundary not predicted

### Baselines

#### 1. Rule-Based Segmenter
**Method**: Heuristic rules for common sandhi patterns
- Add space after visarga (`:`) before consonants
- Add space after danda (`।`, `॥`) punctuation
- Preserve existing word boundaries

**Implementation**: `SimpleSandhiSplitter` with deterministic rules

**Results** (200 test examples):
```
Precision:   1.000
Recall:      1.000
F1 Score:    1.000
Exact Match: 1.000
```

**Note**: Perfect scores because gold set was generated using same splitter. Real gold standard requires manual annotation.

#### 2. Fine-Tuned Token Classification (TODO)
**Method**: BERT-based sequence tagging (BIO scheme)
- Train on silver labels (701 examples)
- Validate on manually annotated gold set
- Tag each character as B (boundary), I (inside), or O (other)

**Expected improvement**: 5-10% over rule-based on real gold data

### Usage

```bash
# Evaluate on gold test set
python scripts/evaluate_sandhi.py \
    --test-file data/benchmarks/sandhi_gold_test.jsonl \
    --output results/sandhi_results.json
```

## Task B: Morphological Acceptability

### Objective
Determine if language models prefer grammatically correct forms over minimal violations.

### Metric

**Acceptability Accuracy**:
```
accuracy = (# pairs where score(grammatical) > score(ungrammatical)) / total_pairs
```

Model is "correct" if it assigns higher score to the grammatical form.

### Scoring Methods

#### Masked Language Models (mBERT, XLM-R, IndicBERT)
**Pseudo-likelihood scoring**:
```python
score(text) = Σ log P(token_i | context_i)
```

For each token, mask it and compute likelihood from the model:
1. Tokenize: text → [t₁, t₂, ..., tₙ]
2. For each tᵢ:
   - Mask position i: [t₁, ..., [MASK], ..., tₙ]
   - Get P(tᵢ | context) from model
   - Add log P(tᵢ) to cumulative score
3. Higher score = more acceptable

**Advantages**:
- No need for special tokens
- Captures bidirectional context
- Standard approach for acceptability judgments

#### Causal Language Models (GPT, etc.)
**Perplexity-based scoring**:
```python
score(text) = -perplexity(text)  # or log P(text)
```

Lower perplexity → more acceptable

### Model Baselines

| Model | Type | Parameters | Coverage |
|-------|------|-----------|----------|
| **mBERT** | Masked LM | 110M | 104 languages (multilingual baseline) |
| **XLM-R Base** | Masked LM | 270M | 100 languages (stronger multilingual) |
| **XLM-R Large** | Masked LM | 550M | 100 languages (high-capacity) |
| **IndicBERT** | Masked LM | 110M | 12 Indic languages (specialized) |
| **MuRIL** | Masked LM | 235M | Indian languages + transliteration |

### Expected Baselines

- **Random**: 50% (coin flip)
- **Frequency-based**: 55-60% (prefer common endings)
- **mBERT**: 60-70% (multilingual pre-training)
- **XLM-R**: 70-80% (stronger multilingual model)
- **IndicBERT**: 75-85% (Indic-specialized, may have Sanskrit data)

### Usage

```bash
# Quick test with 50 pairs
python scripts/evaluate_morphology.py \
    --test-file data/benchmarks/morph_contrast_sets.jsonl \
    --models mbert xlm-r-base \
    --device cpu \
    --sample 50

# Full evaluation (500 pairs)
python scripts/evaluate_morphology.py \
    --test-file data/benchmarks/morph_contrast_sets.jsonl \
    --models mbert xlm-r-base indicbert \
    --device cuda
```

## Unified Benchmark Runner

Run complete evaluation pipeline:

```bash
# Full benchmark with both tasks
python scripts/run_benchmark.py \
    --models mbert xlm-r-base indicbert \
    --device cuda

# Quick test (sandhi + 50 morph pairs)
python scripts/run_benchmark.py \
    --models mbert \
    --sample 50

# Skip sandhi evaluation
python scripts/run_benchmark.py \
    --models mbert xlm-r-base \
    --skip-sandhi
```

**Outputs**:
- `results/sandhi_results_<timestamp>.json`
- `results/morphology_results_<timestamp>.json`
- `results/summary_<timestamp>.csv`

## Results Visualization

Generate plots from evaluation results:

```bash
python scripts/generate_plots.py \
    --sandhi-results results/sandhi_results.json \
    --morph-results results/morphology_results.json \
    --output-dir results/plots
```

**Generated plots**:
1. `sandhi_results.png`: P/R/F1/EM comparison
2. `morphology_overall.png`: Overall accuracy with random baseline
3. `morphology_by_phenomenon.png`: Accuracy breakdown (case vs. number)
4. `combined_results.png`: Side-by-side task comparison

## Results Format

### Summary CSV

```csv
Model,Task,Metric,Score,Details
rule-based,Sandhi Segmentation,Precision,1.000,
rule-based,Sandhi Segmentation,Recall,1.000,
rule-based,Sandhi Segmentation,F1,1.000,
rule-based,Sandhi Segmentation,Exact Match,1.000,
mbert,Morphological Acceptability,Accuracy,0.742,
mbert,Morphological Acceptability,Accuracy (case),0.760,
mbert,Morphological Acceptability,Accuracy (number),0.701,
xlm-r-base,Morphological Acceptability,Accuracy,0.798,
xlm-r-base,Morphological Acceptability,Accuracy (case),0.811,
xlm-r-base,Morphological Acceptability,Accuracy (number),0.769,
```

### JSON Format

**Sandhi results**:
```json
[
  {
    "model": "rule-based",
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "exact_match": 1.0,
    "total_examples": 200
  }
]
```

**Morphology results**:
```json
[
  {
    "model": "mbert",
    "accuracy": 0.742,
    "total_pairs": 500,
    "correct_pairs": 371,
    "by_phenomenon": {
      "case": 0.760,
      "number": 0.701
    },
    "by_stem_class": {
      "a-stem": 0.742
    }
  }
]
```

## Implementation Details

### Metrics Module

**`SandhiMetrics`**: Boundary-level precision/recall/F1
- Extracts boundary positions from segmented text
- Compares predicted vs. gold boundaries
- Handles character-level positions

**`MorphologyMetrics`**: Acceptability accuracy
- Counts correct preferences (gram > ungram)
- Breaks down by phenomenon (case/number)
- Breaks down by stem class (a-stem, etc.)

### Model Wrappers

**Abstract interfaces**:
- `SandhiSegmenter`: `segment(text) → segmented_text`
- `AcceptabilityScorer`: `score(text) → float`

**Implementations**:
- `RuleBasedSegmenter`: Heuristic sandhi splitter
- `MaskedLMScorer`: Pseudo-likelihood for masked LMs
- `ModelFactory`: Create scorers by key (mbert, xlm-r-base, etc.)

### Dependencies

```bash
# Core evaluation (rule-based only)
pip install numpy pandas

# Transformer models
pip install torch transformers

# Visualization
pip install matplotlib
```

## Analysis & Insights

### Expected Findings

1. **Sandhi task**: Rule-based achieves high scores, but requires manual gold standard for real evaluation
2. **Morphology task**: 
   - Case swaps easier than number swaps (more training examples)
   - Larger models (XLM-R Large) outperform smaller (mBERT)
   - Indic-specialized models may excel if Sanskrit in training data

### Error Analysis (TODO)

- Which sandhi rules are hardest to learn?
- Which case pairs cause most confusion? (e.g., dative vs. genitive)
- Do models struggle with dual number (less frequent)?
- Correlation with morpheme frequency in training data?

### Future Extensions

1. **Fine-tuned models**: Train on silver labels, evaluate on gold
2. **Additional models**: GPT-2 multilingual, Gemini, GPT-4 (via API)
3. **Cross-lingual transfer**: Zero-shot from related languages
4. **Probing classifiers**: Linear probes on hidden states
5. **Human evaluation**: Acceptability judgments from Sanskrit scholars

## References

- Salazar et al. (2020). "Masked Language Model Scoring". ACL.
- Warstadt et al. (2019). "Neural Network Acceptability Judgments". TACL.
- Kann et al. (2018). "Verb Inflection in Morphologically Rich Languages". ACL.
- Kakwani et al. (2020). "IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages". EMNLP.

## Next Steps

1. ✓ Implement evaluation infrastructure
2. ✓ Test rule-based sandhi baseline
3. ⚠️ Evaluate transformer models on morphology task
4. ⚠️ Generate results and plots
5. ⚠️ Manually annotate sandhi gold set
6. ⚠️ Write Phase 5 report
