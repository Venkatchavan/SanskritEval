# SanskritEval 🕉️

**Probing Sandhi and Case Generalization in Language Models**

A benchmark suite for evaluating how well language models handle Sanskrit-specific linguistic phenomena, specifically sandhi (phonological fusion at word boundaries) and morphological case agreement.

## 🎯 Project Goals

Sanskrit represents a critical test case for evaluating language model capabilities on:
- **Structurally different low-resource languages**
- **Complex morphological patterns** (8 cases × 3 numbers × 3 genders)
- **Abstract linguistic rules** (sandhi transformations at phonological boundaries)

This benchmark probes whether LMs learn genuine abstraction mechanisms or merely surface-level patterns.

## 📊 Tasks

### Task A: Sandhi Segmentation
**Goal**: Detect word boundaries in phonologically fused Sanskrit strings.

- **Input**: Fused sandhi form (e.g., `rāmo'gacchat`)
- **Output**: Segmented form with boundaries (e.g., `rāmaḥ agacchat`)
- **Metrics**: Precision, Recall, F1 on boundary detection

### Task B: Morphological Acceptability (Contrast Sets)
**Goal**: Test if models have learned case/agreement patterns.

- **Input**: Minimal pairs (grammatical vs. ungrammatical)
  - ✓ Grammatical: correct case ending/agreement
  - ✗ Ungrammatical: incorrect case ending/agreement
- **Scoring**: Log-likelihood or perplexity comparison
- **Metrics**: Accuracy (% pairs where LM prefers grammatical variant)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Venkatchavan/SanskritEval.git
cd SanskritEval

# Create environment
conda env create -f environment.yml
conda activate sanskriteval

# Or use pip
pip install -r requirements.txt
```

### Running the Benchmark

```bash
# Generate sandhi dataset (701 silver + 200 gold)
python scripts/generate_sandhi_data.py

# Generate morphology contrast sets (500 pairs)
python scripts/generate_morph_data.py

# Evaluate sandhi segmentation (rule-based baseline)
python scripts/evaluate_sandhi.py

# Evaluate morphology (transformer models)
python scripts/evaluate_morphology.py --models mbert xlm-r-base --sample 50

# Run complete benchmark
python scripts/run_benchmark.py --models mbert xlm-r-base indicbert

# Generate plots
python scripts/generate_plots.py
```

## 📈 Current Status

### ✅ Completed (Phases 0-4)

- **Data Collection**: 701 verses from Bhagavad Gita
- **Sandhi Dataset**: 701 silver training + 200 gold test examples
- **Morphology Dataset**: 500 contrast pairs (333 case + 167 number)
- **Evaluation Framework**: Metrics, model wrappers, visualization
- **Rule-Based Baseline**: 1.000 F1 on sandhi segmentation

### 📊 Dataset Statistics

| Dataset | Examples | Description |
|---------|----------|-------------|
| Sandhi Silver Train | 701 | Rule-based segmentations |
| Sandhi Gold Test | 200 | Stratified samples (needs manual annotation) |
| Morphology Contrast Sets | 500 | Minimal pairs (case/number perturbations) |
| Source Corpus | 701 verses | Bhagavad Gita |

### 🎯 Models Ready to Evaluate

- **mBERT**: `bert-base-multilingual-cased` (110M params)
- **XLM-R Base**: `xlm-roberta-base` (270M params)
- **XLM-R Large**: `xlm-roberta-large` (550M params)
- **IndicBERT**: `ai4bharat/indic-bert` (110M, Indic languages)
- **MuRIL**: `google/muril-base-cased` (235M, Indian languages)

## 📁 Project Structure

```
sanskriteval/
├── data/
│   ├── raw/                    # 701 verses from Bhagavad Gita
│   └── benchmarks/             # Sandhi (701+200) + Morphology (500)
├── src/sanskriteval/
│   ├── data/                   # Data generation (normalizer, sandhi, morphology)
│   ├── models/                 # Model wrappers (rule-based, transformers)
│   ├── metrics/                # Evaluation metrics (sandhi, morphology)
│   └── utils/                  # Text processing, config, logging
├── scripts/
│   ├── generate_*.py           # Dataset generation
│   ├── evaluate_*.py           # Task evaluation
│   ├── run_benchmark.py        # Unified benchmark runner
│   └── generate_plots.py       # Results visualization
├── docs/                       # Phase documentation
├── results/                    # Evaluation outputs (JSON, CSV, plots)
└── README.md
```

## 📦 Deliverables

- ✅ **Benchmark Dataset**: 
  - Sandhi: 701 training + 200 test (JSONL)
  - Morphology: 500 contrast pairs (JSONL)
  - Generation scripts included
- ✅ **Evaluation Pipeline**: 
  - Metrics: P/R/F1 for sandhi, acceptability accuracy for morphology
  - Model wrappers: Rule-based + 5 transformer models
  - Unified runner with CSV summary
- ✅ **Visualization**: 4 plot types (sandhi, morphology overall/breakdown, combined)
- ⚠️ **Results**: Models × Tasks comparison (in progress)
- ⚠️ **Report**: 6-8 page paper-style documentation (Phase 5)

## 🔬 Baseline Results

### Sandhi Segmentation (Rule-Based)

| Model | Precision | Recall | F1 | Exact Match |
|-------|-----------|--------|-------|-------------|
| Rule-Based | 1.000 | 1.000 | 1.000 | 1.000 |

*Note*: Gold set was generated using same splitter; manual annotation needed for real evaluation.

### Morphological Acceptability (Expected)

| Model | Expected Accuracy | Notes |
|-------|------------------|-------|
| Random | 50% | Baseline |
| mBERT | 60-70% | Multilingual |
| XLM-R Base | 70-80% | Stronger multilingual |
| IndicBERT | 75-85% | Indic-specialized |

## 📚 Citation

If you use this benchmark, please cite:

```bibtex
@misc{sanskriteval2025,
  title={SanskritEval: Probing Sandhi and Case Generalization in Language Models},
  author={[Your Name]},
  year={2025},
  url={https://github.com/Venkatchavan/SanskritEval}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- GitHub: [@Venkatchavan](https://github.com/Venkatchavan)
- Issues: [GitHub Issues](https://github.com/Venkatchavan/SanskritEval/issues)

---

**Note**: This is an active research project. Dataset and evaluation scripts will be continuously updated.
