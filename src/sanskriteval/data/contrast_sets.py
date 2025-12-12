"""
Generate morphological contrast sets (minimal pairs) for evaluating
language models' understanding of Sanskrit case and number inflection.
"""

import re
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
from pathlib import Path

from .morphology import (
    SanskritDeclensions,
    DeclensionPattern,
    MorphologyPerturbation,
    Case, Number, Gender,
    generate_case_perturbations,
    generate_number_perturbations
)


@dataclass
class ContrastPair:
    """A minimal pair for morphological evaluation."""
    id: str
    phenomenon: str  # "case" or "number"
    grammatical: str  # Correct form
    ungrammatical: str  # Incorrect form (with perturbation)
    stem: str  # Word stem
    context: str  # Original sentence/phrase (optional)
    metadata: Dict[str, str]  # Additional info
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON."""
        return asdict(self)


class NounExtractor:
    """Extract nouns with recognizable endings from Sanskrit text."""
    
    def __init__(self):
        self.declensions = SanskritDeclensions()
        self.patterns = self.declensions.get_patterns()
    
    def extract_nouns_from_verse(self, verse: str) -> List[Tuple[str, str]]:
        """Extract nouns with identifiable endings.
        
        Args:
            verse: Sanskrit verse
            
        Returns:
            List of (word, ending) tuples
        """
        # Split into words
        words = re.findall(r'[\u0900-\u097F]+', verse)
        
        noun_candidates = []
        
        for word in words:
            if len(word) < 3:
                continue
            
            # Try to match endings from our patterns
            for pattern in self.patterns:
                for ending in set(pattern.endings.values()):
                    if ending and word.endswith(ending):
                        # Found a potential noun
                        stem = word[:-len(ending)] if ending else word
                        noun_candidates.append((word, ending, stem, pattern))
                        break
        
        return noun_candidates
    
    def extract_from_corpus(self, corpus_path: Path, max_nouns: int = 500) -> List[Tuple[str, str, DeclensionPattern]]:
        """Extract nouns from entire corpus.
        
        Args:
            corpus_path: Path to corpus file (one verse per line)
            max_nouns: Maximum number of unique nouns to extract
            
        Returns:
            List of (word, stem, pattern) tuples
        """
        nouns = []
        seen_stems = set()
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove verse IDs
                text = re.sub(r'^\d+\.\d+:\s*', '', line.strip())
                
                candidates = self.extract_nouns_from_verse(text)
                
                for word, ending, stem, pattern in candidates:
                    if stem not in seen_stems and len(stem) >= 2:
                        nouns.append((word, stem, pattern))
                        seen_stems.add(stem)
                        
                        if len(nouns) >= max_nouns:
                            return nouns
        
        return nouns


class ContrastSetGenerator:
    """Generate minimal pairs for morphological evaluation."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        self.declensions = SanskritDeclensions()
    
    def generate_case_pair(
        self,
        stem: str,
        pattern: DeclensionPattern,
        perturbation: MorphologyPerturbation,
        pair_id: str
    ) -> Optional[ContrastPair]:
        """Generate a case swap minimal pair.
        
        Args:
            stem: Word stem
            pattern: Declension pattern
            perturbation: Case perturbation to apply
            pair_id: Unique ID for this pair
            
        Returns:
            ContrastPair or None if generation fails
        """
        if perturbation.phenomenon != "case_swap":
            return None
        
        # Generate grammatical form
        grammatical = stem + perturbation.from_ending
        
        # Generate ungrammatical form (wrong case)
        ungrammatical = stem + perturbation.to_ending
        
        return ContrastPair(
            id=pair_id,
            phenomenon="case",
            grammatical=grammatical,
            ungrammatical=ungrammatical,
            stem=stem,
            context="",  # Could add sentence context later
            metadata={
                "stem_class": pattern.stem_class,
                "gender": pattern.gender.value,
                "number": perturbation.from_number.value,
                "correct_case": perturbation.from_case.value,
                "incorrect_case": perturbation.to_case.value,
                "ending_from": perturbation.from_ending,
                "ending_to": perturbation.to_ending
            }
        )
    
    def generate_number_pair(
        self,
        stem: str,
        pattern: DeclensionPattern,
        perturbation: MorphologyPerturbation,
        pair_id: str
    ) -> Optional[ContrastPair]:
        """Generate a number swap minimal pair.
        
        Args:
            stem: Word stem
            pattern: Declension pattern
            perturbation: Number perturbation to apply
            pair_id: Unique ID for this pair
            
        Returns:
            ContrastPair or None if generation fails
        """
        if perturbation.phenomenon != "number_swap":
            return None
        
        # Generate grammatical form
        grammatical = stem + perturbation.from_ending
        
        # Generate ungrammatical form (wrong number)
        ungrammatical = stem + perturbation.to_ending
        
        return ContrastPair(
            id=pair_id,
            phenomenon="number",
            grammatical=grammatical,
            ungrammatical=ungrammatical,
            stem=stem,
            context="",
            metadata={
                "stem_class": pattern.stem_class,
                "gender": pattern.gender.value,
                "case": perturbation.from_case.value,
                "correct_number": perturbation.from_number.value,
                "incorrect_number": perturbation.to_number.value,
                "ending_from": perturbation.from_ending,
                "ending_to": perturbation.to_ending
            }
        )
    
    def generate_pairs_for_stem(
        self,
        stem: str,
        pattern: DeclensionPattern,
        num_case_pairs: int = 3,
        num_number_pairs: int = 2
    ) -> List[ContrastPair]:
        """Generate multiple minimal pairs for a single stem.
        
        Args:
            stem: Word stem
            pattern: Declension pattern
            num_case_pairs: Number of case perturbations to generate
            num_number_pairs: Number of number perturbations to generate
            
        Returns:
            List of ContrastPair objects
        """
        pairs = []
        
        # Generate case perturbations (keeping number constant)
        case_perts = generate_case_perturbations(pattern, Number.SINGULAR)
        selected_case = random.sample(
            case_perts,
            min(num_case_pairs, len(case_perts))
        )
        
        for i, pert in enumerate(selected_case):
            pair = self.generate_case_pair(
                stem, pattern, pert,
                pair_id=f"{stem}_case_{i+1}"
            )
            if pair:
                pairs.append(pair)
        
        # Generate number perturbations (keeping case constant)
        number_perts = generate_number_perturbations(pattern, Case.NOMINATIVE)
        selected_number = random.sample(
            number_perts,
            min(num_number_pairs, len(number_perts))
        )
        
        for i, pert in enumerate(selected_number):
            pair = self.generate_number_pair(
                stem, pattern, pert,
                pair_id=f"{stem}_number_{i+1}"
            )
            if pair:
                pairs.append(pair)
        
        return pairs
    
    def generate_dataset(
        self,
        nouns: List[Tuple[str, str, DeclensionPattern]],
        target_size: int = 500
    ) -> List[ContrastPair]:
        """Generate full contrast set dataset.
        
        Args:
            nouns: List of (word, stem, pattern) tuples
            target_size: Target number of pairs
            
        Returns:
            List of ContrastPair objects
        """
        all_pairs = []
        
        # Shuffle nouns for variety
        shuffled_nouns = nouns.copy()
        random.shuffle(shuffled_nouns)
        
        pairs_per_stem = max(1, target_size // len(nouns))
        
        for word, stem, pattern in shuffled_nouns:
            pairs = self.generate_pairs_for_stem(
                stem, pattern,
                num_case_pairs=pairs_per_stem,
                num_number_pairs=pairs_per_stem // 2
            )
            all_pairs.extend(pairs)
            
            if len(all_pairs) >= target_size:
                break
        
        # Shuffle final dataset
        random.shuffle(all_pairs)
        
        # Assign sequential IDs
        for i, pair in enumerate(all_pairs[:target_size], 1):
            pair.id = f"morph_{i:04d}"
        
        return all_pairs[:target_size]
