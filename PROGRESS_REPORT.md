# SanskritEval Progress Report

## Project Overview
**Repository**: https://github.com/Venkatchavan/SanskritEval  
**Objective**: Build benchmark for probing Sanskrit linguistic phenomena in language models  
**Tasks**: Sandhi segmentation + Morphological acceptability

## Completed Phases

### ✓ Phase 0: Repository Setup (Days 1)
- Full project structure with `src/`, `data/`, `scripts/`, `docs/`
- Package management: `requirements.txt`, `environment.yml`, `pyproject.toml`
- Development tools: `Makefile`, `.gitignore`, `config.yaml`
- Git initialization and GitHub push
- MIT License

### ✓ Phase 1: Data Normalization (Day 2)
- Text processing utilities:
  - Unicode normalization (NFC/NFD)
  - Punctuation cleaning
  - Script conversion support (Devanagari ↔ IAST)
  - Verse ID extraction
- `DataNormalizer` class for corpus processing
- Test scripts and validation

### ✓ Phase 2: Sandhi Dataset (Days 3-5)
- Rule-based sandhi splitter:
  - Visarga rules (`:` → `स्`, `र्`)
  - Vowel sandhi (अ-अ → आ, अ-इ → ए, etc.)
  - Consonant sandhi (त् → द्, etc.)
- Dataset expansion:
  - Initial: 10 sample verses
  - **Final: 701 verses** from Bhagavad Gita CSV
  - Converted CSV → clean text format
- Generated datasets:
  - **Silver training**: 701 examples (confidence: 0.5-0.6)
  - **Gold test**: 200 examples (stratified by chapter, needs manual verification)
- Format: JSONL with `{verse_id, fused, segmented, confidence, is_gold}`
- Documentation: `docs/PHASE2_SANDHI.md`, `data/benchmarks/ANNOTATION_GUIDE.md`

### ✓ Phase 3: Morphological Contrast Sets (Days 6-7)
- Declension pattern system:
  - **8 cases**: nominative, accusative, instrumental, dative, ablative, genitive, locative, vocative
  - **3 numbers**: singular, dual, plural
  - **3 genders**: masculine, neuter, feminine (a-stems)
- Noun extraction pipeline:
  - Pattern matching for common endings
  - Stem extraction and deduplication
  - **200 unique stems** extracted from 701 verses
- Contrast pair generation:
  - **Case perturbations**: Keep number constant, swap case endings
  - **Number perturbations**: Keep case constant, swap number endings
  - Minimal differences for precise evaluation
- **Generated dataset**: 500 pairs (333 case + 167 number)
- Format: JSONL with `{id, phenomenon, grammatical, ungrammatical, stem, context, metadata}`
- Documentation: `docs/PHASE3_MORPHOLOGY.md`

## Current Statistics

### Data Assets
| Asset | Count | Description |
|-------|-------|-------------|
| Source verses | 701 | Bhagavad Gita corpus |
| Sandhi silver training | 701 | Rule-based segmentations |
| Sandhi gold test | 200 | Stratified samples (needs annotation) |
| Morphology contrast pairs | 500 | Minimal pairs (333 case + 167 number) |
| Unique noun stems | 200 | Extracted from corpus |

### Code Modules
| Module | Purpose | Status |
|--------|---------|--------|
| `utils/text_processing.py` | Unicode, cleaning, script conversion | ✓ Complete |
| `data/normalizer.py` | Corpus normalization pipeline | ✓ Complete |
| `data/sandhi.py` | Rule-based sandhi splitting | ✓ Complete |
| `data/morphology.py` | Declension patterns (8 cases × 3 numbers) | ✓ Complete |
| `data/contrast_sets.py` | Noun extraction, pair generation | ✓ Complete |
| `scripts/normalize_data.py` | Text normalization CLI | ✓ Complete |
| `scripts/convert_gita_csv.py` | CSV → text converter | ✓ Complete |
| `scripts/generate_sandhi_data.py` | Sandhi dataset generation | ✓ Complete |
| `scripts/generate_morph_data.py` | Morphology dataset generation | ✓ Complete |

## Next Steps

### Phase 4: Model Evaluation (Days 8-10)
- [ ] Implement evaluation pipeline
  - Sandhi: Precision/Recall/F1 for boundary detection
  - Morphology: Acceptability accuracy (% correct preferences)
- [ ] Select models to evaluate:
  - **Baseline**: Random (50%), Frequency-based
  - **Multilingual**: mBERT, XLM-R, IndicBERT
  - **Generative**: GPT-2 (multilingual), Gemini, GPT-4
- [ ] Run evaluations and collect results
- [ ] Analyze error patterns:
  - Which sandhi rules are hardest?
  - Which cases/numbers cause most confusion?
  - Correlation with training data frequency?

### Phase 5: Results & Reporting (Days 11-12)
- [ ] Generate results tables (models × tasks)
- [ ] Create visualization plots:
  - Model comparison bar charts
  - Error breakdown by phenomenon
  - Confusion matrices for cases/numbers
- [ ] Write paper-style report (6-8 pages):
  - Abstract & Introduction
  - Related Work
  - Dataset Construction (Phases 1-3)
  - Evaluation Protocol
  - Results & Analysis
  - Conclusion & Future Work
- [ ] Create README examples and usage guide

### Manual Annotation
- [ ] Annotate 200 sandhi gold test examples
  - Verify rule-based segmentations
  - Correct errors, mark edge cases
  - Update confidence scores

## Repository Structure (Current)

```
SanskritEval/
├── data/
│   ├── raw/
│   │   ├── gita_full_700.txt        # 701 verses
│   │   ├── Bhagwad_Gita.csv         # Source CSV
│   │   └── DATA_SOURCES.md
│   └── benchmarks/
│       ├── sandhi_silver_train.jsonl    # 701 examples
│       ├── sandhi_gold_test.jsonl       # 200 examples
│       ├── morph_contrast_sets.jsonl    # 500 pairs
│       └── ANNOTATION_GUIDE.md
├── src/sanskriteval/
│   ├── data/
│   │   ├── normalizer.py
│   │   ├── sandhi.py
│   │   ├── morphology.py
│   │   └── contrast_sets.py
│   ├── utils/
│   │   ├── text_processing.py
│   │   ├── config.py
│   │   └── logging.py
│   ├── models/           # (placeholder)
│   └── metrics/          # (placeholder)
├── scripts/
│   ├── normalize_data.py
│   ├── convert_gita_csv.py
│   ├── generate_sandhi_data.py
│   └── generate_morph_data.py
├── docs/
│   ├── PHASE2_SANDHI.md
│   └── PHASE3_MORPHOLOGY.md
├── requirements.txt
├── Makefile
└── README.md
```

## Key Achievements

1. **Reproducible pipeline**: All datasets generated from scripts
2. **Comprehensive corpus**: Full 701 verses from Bhagavad Gita
3. **Dual-task benchmark**: Both sandhi (syntax) and morphology (grammar)
4. **Minimal pairs methodology**: Precise evaluation with controlled perturbations
5. **Detailed metadata**: Rich annotations for error analysis
6. **Professional structure**: Package layout, documentation, version control

## Timeline Summary

- **Days 1-2**: Foundation (repo + normalization)
- **Days 3-5**: Sandhi task implementation
- **Days 6-7**: Morphology task implementation
- **Days 8-10**: Model evaluation (next)
- **Days 11-12**: Results & reporting (next)

## Contact & Links

- **GitHub**: https://github.com/Venkatchavan/SanskritEval
- **License**: MIT
- **Python**: 3.10+
- **Status**: Phase 3 complete, ready for evaluation phase
