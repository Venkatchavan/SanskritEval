"""
Sandhi (phonological fusion) rules and utilities for Sanskrit.

This module implements common sandhi rules for automatic segmentation.
These are "silver" labels - may not be 100% accurate but provide good starting points.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SandhiRule:
    """A sandhi transformation rule."""
    pattern: str
    split_form: str
    description: str
    confidence: float = 0.7  # Confidence score (0-1)


class SandhiRules:
    """Collection of common sandhi rules for boundary detection."""
    
    # Common sandhi patterns (simplified - for silver labels)
    VOWEL_SANDHI_RULES = [
        # a + a = ā
        SandhiRule(r'ा([aāiīuūṛṝeaioau])', ' ā \\1', 'a + a = ā', 0.8),
        
        # Visarga sandhi
        SandhiRule(r'ः\s*([kpśs])', 'ḥ \\1', 'visarga before voiceless', 0.7),
        
        # Common word boundary markers
        SandhiRule(r'्([क-ह])', '् \\1', 'halanta + consonant', 0.6),
    ]
    
    COMMON_SPLITS = [
        # Common compound patterns (Devanagari)
        ('वाश्च', 'वाः च'),
        ('स्त्व', 'स्त् त्व'),
        ('तस्य', 'तस्य'),  # Often not split
        ('एव', 'एव'),  # Often not split
    ]
    
    @classmethod
    def detect_potential_boundaries(cls, text: str) -> List[int]:
        """Detect potential word boundaries using heuristics.
        
        Args:
            text: Sanskrit text (Devanagari or IAST)
            
        Returns:
            List of character positions that might be word boundaries
        """
        boundaries = []
        
        # Method 1: Space-based (if already partially split)
        for i, char in enumerate(text):
            if char == ' ':
                boundaries.append(i)
        
        # Method 2: Common markers
        # Danda marks (।)
        for match in re.finditer(r'[।॥]', text):
            boundaries.append(match.start())
        
        # Method 3: Visarga followed by consonant (often word boundary)
        for match in re.finditer(r'[ः](?=[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])', text):
            boundaries.append(match.end())
        
        # Method 4: Explicit halant (virama) often marks boundary
        for match in re.finditer(r'्\s*(?=[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])', text):
            boundaries.append(match.end())
        
        return sorted(set(boundaries))


@dataclass
class SandhiExample:
    """A sandhi segmentation example."""
    fused: str  # Fused form (with sandhi)
    segmented: str  # Segmented form (word boundaries marked)
    verse_id: str
    confidence: float = 0.5  # Confidence in segmentation (0-1)
    is_gold: bool = False  # True if manually verified
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'verse_id': self.verse_id,
            'fused': self.fused,
            'segmented': self.segmented,
            'confidence': self.confidence,
            'is_gold': self.is_gold
        }


class SimpleSandhiSplitter:
    """Simple rule-based sandhi splitter for silver label generation."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize splitter.
        
        Args:
            confidence_threshold: Minimum confidence for suggesting a split
        """
        self.confidence_threshold = confidence_threshold
        self.rules = SandhiRules()
    
    def split_verse(
        self,
        verse: str,
        verse_id: str = "unknown"
    ) -> SandhiExample:
        """Attempt to split a verse at sandhi boundaries.
        
        Args:
            verse: Fused Sanskrit verse
            verse_id: Verse identifier
            
        Returns:
            SandhiExample with suggested segmentation
        """
        # For now, use simple heuristics:
        # 1. Keep existing spaces
        # 2. Add space after visarga before consonant
        # 3. Add space after danda
        
        segmented = verse
        
        # Normalize spaces
        segmented = re.sub(r'\s+', ' ', segmented)
        
        # Add space after danda if not present
        segmented = re.sub(r'([।॥])(?=[^\s])', r'\1 ', segmented)
        
        # Add space after visarga before consonant (conservative)
        # This is a common word boundary in Sanskrit
        segmented = re.sub(
            r'(ः)(?=[कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह])',
            r'\1 ',
            segmented
        )
        
        # Clean up multiple spaces
        segmented = re.sub(r'\s+', ' ', segmented).strip()
        
        # Calculate confidence based on how much we changed
        confidence = 0.5  # Base confidence for silver labels
        if segmented != verse:
            confidence = 0.6  # Slightly higher if we made changes
        
        return SandhiExample(
            fused=verse,
            segmented=segmented,
            verse_id=verse_id,
            confidence=confidence,
            is_gold=False
        )
    
    def segment(self, text: str) -> str:
        """Simple interface for segmentation (used by evaluation).
        
        Args:
            text: Fused Sanskrit text
            
        Returns:
            Segmented text with spaces
        """
        example = self.split_verse(text)
        return example.segmented

    
    def mark_boundaries(self, text: str) -> List[Tuple[int, float]]:
        """Mark potential word boundaries with confidence scores.
        
        Args:
            text: Sanskrit text
            
        Returns:
            List of (position, confidence) tuples
        """
        boundaries = []
        
        # Detect using heuristics
        positions = self.rules.detect_potential_boundaries(text)
        
        for pos in positions:
            # Assign confidence based on marker type
            char = text[pos] if pos < len(text) else ''
            if char in '।॥':
                confidence = 0.9  # High confidence for danda
            elif char == ' ':
                confidence = 0.8  # High for existing spaces
            else:
                confidence = 0.6  # Medium for inferred boundaries
            
            boundaries.append((pos, confidence))
        
        return boundaries


def create_manual_gold_template(
    silver_examples: List[SandhiExample],
    num_samples: int = 200
) -> List[SandhiExample]:
    """Create template for manual gold annotation.
    
    Args:
        silver_examples: Silver label examples
        num_samples: Number of samples for gold set
        
    Returns:
        List of examples to be manually corrected
    """
    # Sample evenly across chapters if possible
    # For now, just take first N examples
    sampled = silver_examples[:num_samples]
    
    # Mark for manual review
    for example in sampled:
        example.confidence = 0.0  # Mark as needing verification
        example.is_gold = False  # Will be true after manual correction
    
    return sampled
