# SanskritEval

Benchmark suite for evaluating language models on Sanskrit linguistic phenomena (sandhi segmentation and morphological case agreement).

**Status**: Active Development 🚧

## Repository: https://github.com/Venkatchavan/SanskritEval.git

For full documentation, see [README.md](README.md)

## Quick Setup

```bash
# Clone
git clone https://github.com/Venkatchavan/SanskritEval.git
cd SanskritEval

# Install
conda env create -f environment.yml
conda activate sanskriteval

# Generate data
make data

# Evaluate
make evaluate

# Report
make report
```

## Structure

- `data/` - Raw, processed, and benchmark datasets
- `src/sanskriteval/` - Main package (data, models, metrics, utils)
- `scripts/` - Standalone generation and evaluation scripts
- `notebooks/` - Exploratory Jupyter notebooks
- `reports/` - Generated results and visualizations

## License

MIT - See [LICENSE](LICENSE)
