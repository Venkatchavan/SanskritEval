"""Data generation and processing modules."""

from .normalizer import DataNormalizer, NormalizedVerse
from .sandhi import SimpleSandhiSplitter, SandhiExample, SandhiRules
from .morphology import (
    Case, Number, Gender,
    DeclensionPattern, SanskritDeclensions,
    generate_case_perturbations, generate_number_perturbations
)
from .contrast_sets import ContrastPair, NounExtractor, ContrastSetGenerator

__all__ = [
    "DataNormalizer",
    "NormalizedVerse",
    "SimpleSandhiSplitter",
    "SandhiExample",
    "SandhiRules",
    "Case",
    "Number",
    "Gender",
    "DeclensionPattern",
    "SanskritDeclensions",
    "generate_case_perturbations",
    "generate_number_perturbations",
    "ContrastPair",
    "NounExtractor",
    "ContrastSetGenerator"
]
