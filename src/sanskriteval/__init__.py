"""
SanskritEval: Probing Sandhi and Case Generalization in Language Models

A benchmark suite for evaluating language model capabilities on Sanskrit-specific
linguistic phenomena.
"""

__version__ = "0.1.0"
__author__ = "Venkatchavan"
__email__ = ""

from . import data
from . import models
from . import metrics
from . import utils

__all__ = ["data", "models", "metrics", "utils"]
