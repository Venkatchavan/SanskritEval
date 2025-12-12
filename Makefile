.PHONY: help install clean data evaluate report all test lint format

# Default target
help:
	@echo "SanskritEval - Makefile Commands"
	@echo "================================="
	@echo "setup          - Create conda environment and install dependencies"
	@echo "install        - Install package in development mode"
	@echo "clean          - Remove generated files and caches"
	@echo "data           - Generate all benchmark datasets"
	@echo "data-sandhi    - Generate sandhi segmentation dataset"
	@echo "data-morph     - Generate morphological contrast sets"
	@echo "evaluate       - Run evaluation on all models"
	@echo "eval-sandhi    - Evaluate sandhi task only"
	@echo "eval-morph     - Evaluate morphology task only"
	@echo "report         - Generate results tables and plots"
	@echo "test           - Run unit tests"
	@echo "lint           - Run linting checks"
	@echo "format         - Format code with black"
	@echo "all            - Run complete pipeline (data + eval + report)"

# Environment setup
setup:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "Environment created! Activate with: conda activate sanskriteval"

install:
	@echo "Installing package in development mode..."
	pip install -e .

# Clean up
clean:
	@echo "Cleaning up generated files..."
	rm -rf data/processed/*
	rm -rf data/benchmarks/*
	rm -rf reports/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Data generation
data: data-sandhi data-morph

data-sandhi:
	@echo "Generating sandhi segmentation dataset..."
	python scripts/generate_sandhi_data.py

data-morph:
	@echo "Generating morphological contrast sets..."
	python scripts/generate_morph_data.py

# Evaluation
evaluate: eval-sandhi eval-morph

eval-sandhi:
	@echo "Evaluating sandhi segmentation task..."
	python scripts/evaluate_sandhi.py

eval-morph:
	@echo "Evaluating morphological acceptability task..."
	python scripts/evaluate_morph.py

# Reporting
report:
	@echo "Generating results report..."
	python scripts/generate_report.py
	@echo "Report saved to reports/"

# Complete pipeline
all: clean data evaluate report
	@echo "Pipeline complete! Check reports/ for results."

# Development tools
test:
	@echo "Running tests..."
	pytest tests/ -v

lint:
	@echo "Running linting checks..."
	black --check src/ scripts/
	@echo "Linting complete!"

format:
	@echo "Formatting code..."
	black src/ scripts/ tests/
	@echo "Formatting complete!"

# Quick verification
verify:
	@echo "Verifying project structure..."
	@test -d data/raw || (echo "✗ data/raw missing" && exit 1)
	@test -d data/processed || (echo "✗ data/processed missing" && exit 1)
	@test -d data/benchmarks || (echo "✗ data/benchmarks missing" && exit 1)
	@test -d src/sanskriteval || (echo "✗ src/sanskriteval missing" && exit 1)
	@test -d scripts || (echo "✗ scripts missing" && exit 1)
	@test -d notebooks || (echo "✗ notebooks missing" && exit 1)
	@test -d reports || (echo "✗ reports missing" && exit 1)
	@test -f requirements.txt || (echo "✗ requirements.txt missing" && exit 1)
	@test -f environment.yml || (echo "✗ environment.yml missing" && exit 1)
	@echo "✓ Project structure verified!"
