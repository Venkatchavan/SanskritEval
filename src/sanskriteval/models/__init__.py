"""Model wrappers for evaluation."""

from .base import SandhiSegmenter, AcceptabilityScorer
from .rule_based import RuleBasedSegmenter

try:
    from .transformers_models import MaskedLMScorer, ModelFactory
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    MaskedLMScorer = None
    ModelFactory = None

try:
    from .probing import (
        LinearProbe,
        LayerWiseProber,
        prepare_sandhi_probe_data,
        prepare_morphology_probe_data
    )
    PROBING_AVAILABLE = True
except ImportError:
    PROBING_AVAILABLE = False
    LinearProbe = None
    LayerWiseProber = None

__all__ = [
    "SandhiSegmenter",
    "AcceptabilityScorer",
    "RuleBasedSegmenter",
    "MaskedLMScorer",
    "ModelFactory",
    "TRANSFORMERS_AVAILABLE",
    "LinearProbe",
    "LayerWiseProber",
    "prepare_sandhi_probe_data",
    "prepare_morphology_probe_data",
    "PROBING_AVAILABLE"
]
