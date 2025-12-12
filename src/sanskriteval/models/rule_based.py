"""Rule-based baseline models."""

from typing import List
from .base import SandhiSegmenter
from ..data.sandhi import SimpleSandhiSplitter


class RuleBasedSegmenter(SandhiSegmenter):
    """Rule-based sandhi segmentation baseline.
    
    Uses SimpleSandhiSplitter with deterministic rules.
    """
    
    def __init__(self):
        """Initialize rule-based segmenter."""
        self.splitter = SimpleSandhiSplitter()
    
    def segment(self, fused_text: str) -> str:
        """Segment using rule-based splitter.
        
        Args:
            fused_text: Fused Sanskrit text
            
        Returns:
            Segmented text with spaces
        """
        return self.splitter.segment(fused_text)
    
    def segment_batch(self, fused_texts: List[str]) -> List[str]:
        """Segment batch of texts.
        
        Args:
            fused_texts: List of fused texts
            
        Returns:
            List of segmented texts
        """
        return [self.segment(text) for text in fused_texts]
