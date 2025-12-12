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
# Generate benchmark datasets
make generate-data

# Run evaluation on all models
make evaluate

# Generate report and plots
make report
```

## 📁 Project Structure

```
sanskriteval/
├── data/
│   ├── raw/              # Raw Sanskrit corpora
│   ├── processed/        # Cleaned and preprocessed data
│   └── benchmarks/       # Final benchmark datasets (JSONL/CSV)
├── src/
│   └── sanskriteval/     # Main package
│       ├── data/         # Data generation and processing
│       ├── models/       # Model wrappers and evaluation
│       ├── metrics/      # Evaluation metrics
│       └── utils/        # Utilities
├── scripts/              # Standalone scripts for data/eval
├── notebooks/            # Jupyter notebooks for exploration
├── reports/              # Generated reports and plots
├── Makefile              # Automation commands
└── README.md
```

## 📦 Deliverables

- ✅ **Benchmark Dataset**: JSONL/CSV format with generation scripts
- ✅ **Evaluation Pipeline**: Reproducible model evaluation framework
- ✅ **Results**: Models × Tasks comparison table + visualizations
- ✅ **Report**: 6-8 page paper-style documentation
- 🎯 **Optional**: Zenodo DOI for dataset release

## 🔬 Evaluated Models

- GPT-3.5/GPT-4 (OpenAI)
- Claude (Anthropic)
- Gemini (Google)
- LLaMA variants
- Multilingual models (mBERT, XLM-R)
- Sanskrit-specific models (if available)

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
