"""Base classes for model wrappers."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class SandhiSegmenter(ABC):
    """Abstract base class for sandhi segmentation models."""
    
    @abstractmethod
    def segment(self, fused_text: str) -> str:
        """Segment fused text into morphemes.
        
        Args:
            fused_text: Sanskrit text with sandhi applied
            
        Returns:
            Segmented text with spaces or '+' between morphemes
        """
        pass
    
    @abstractmethod
    def segment_batch(self, fused_texts: List[str]) -> List[str]:
        """Segment batch of fused texts.
        
        Args:
            fused_texts: List of fused Sanskrit texts
            
        Returns:
            List of segmented texts
        """
        pass


class AcceptabilityScorer(ABC):
    """Abstract base class for morphological acceptability scoring."""
    
    @abstractmethod
    def score(self, text: str) -> float:
        """Score a text for acceptability/likelihood.
        
        For masked LMs: Use pseudo-likelihood (sum of log p(token_i | context))
        For causal LMs: Use negative perplexity or log probability
        
        Higher score = more acceptable/likely
        
        Args:
            text: Sanskrit text to score
            
        Returns:
            Acceptability score (higher = more acceptable)
        """
        pass
    
    @abstractmethod
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score batch of texts.
        
        Args:
            texts: List of Sanskrit texts
            
        Returns:
            List of acceptability scores
        """
        pass
    
    def compare_pair(self, grammatical: str, ungrammatical: str) -> bool:
        """Compare grammatical vs ungrammatical form.
        
        Args:
            grammatical: Correct form
            ungrammatical: Incorrect form
            
        Returns:
            True if model prefers grammatical form
        """
        score_gram = self.score(grammatical)
        score_ungram = self.score(ungrammatical)
        return score_gram > score_ungram
