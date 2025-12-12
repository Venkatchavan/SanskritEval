"""
Sanskrit morphology rules for generating contrast sets.

This module implements simplified Sanskrit declension patterns
for generating grammatical/ungrammatical minimal pairs.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Case(Enum):
    """Sanskrit cases (vibhakti)."""
    NOMINATIVE = "nominative"      # प्रथमा (kartā)
    ACCUSATIVE = "accusative"      # द्वितीया (karma)
    INSTRUMENTAL = "instrumental"  # तृतीया (karaṇa)
    DATIVE = "dative"             # चतुर्थी (sampradāna)
    ABLATIVE = "ablative"         # पञ्चमी (apādāna)
    GENITIVE = "genitive"         # षष्ठी (sambandha)
    LOCATIVE = "locative"         # सप्तमी (adhikaraṇa)
    VOCATIVE = "vocative"         # संबोधन (āmantrita)


class Number(Enum):
    """Sanskrit numbers (vachana)."""
    SINGULAR = "singular"  # एकवचन
    DUAL = "dual"         # द्विवचन
    PLURAL = "plural"     # बहुवचन


class Gender(Enum):
    """Sanskrit genders (linga)."""
    MASCULINE = "masculine"  # पुंलिङ्ग
    FEMININE = "feminine"    # स्त्रीलिङ्ग
    NEUTER = "neuter"       # नपुंसकलिङ्ग


@dataclass
class DeclensionPattern:
    """A declension pattern for a noun class."""
    stem_class: str  # e.g., "a-stem", "i-stem"
    gender: Gender
    endings: Dict[Tuple[Case, Number], str]  # (case, number) -> ending


class SanskritDeclensions:
    """Collection of Sanskrit declension patterns."""
    
    # Masculine a-stems (like rāma, putra)
    MASCULINE_A_STEM = DeclensionPattern(
        stem_class="a-stem",
        gender=Gender.MASCULINE,
        endings={
            # Singular
            (Case.NOMINATIVE, Number.SINGULAR): "ः",  # rāmaḥ
            (Case.ACCUSATIVE, Number.SINGULAR): "म्",  # rāmam
            (Case.INSTRUMENTAL, Number.SINGULAR): "एन",  # rāmeṇa
            (Case.DATIVE, Number.SINGULAR): "आय",  # rāmāya
            (Case.ABLATIVE, Number.SINGULAR): "आत्",  # rāmāt
            (Case.GENITIVE, Number.SINGULAR): "अस्य",  # rāmasya
            (Case.LOCATIVE, Number.SINGULAR): "ए",  # rāme
            (Case.VOCATIVE, Number.SINGULAR): "",  # rāma
            
            # Dual
            (Case.NOMINATIVE, Number.DUAL): "औ",  # rāmau
            (Case.ACCUSATIVE, Number.DUAL): "औ",
            (Case.INSTRUMENTAL, Number.DUAL): "आभ्याम्",  # rāmābhyām
            (Case.DATIVE, Number.DUAL): "आभ्याम्",
            (Case.ABLATIVE, Number.DUAL): "आभ्याम्",
            (Case.GENITIVE, Number.DUAL): "अयोः",  # rāmayoḥ
            (Case.LOCATIVE, Number.DUAL): "अयोः",
            (Case.VOCATIVE, Number.DUAL): "औ",
            
            # Plural
            (Case.NOMINATIVE, Number.PLURAL): "आः",  # rāmāḥ
            (Case.ACCUSATIVE, Number.PLURAL): "आन्",  # rāmān
            (Case.INSTRUMENTAL, Number.PLURAL): "ऐः",  # rāmaiḥ
            (Case.DATIVE, Number.PLURAL): "एभ्यः",  # rāmebhyaḥ
            (Case.ABLATIVE, Number.PLURAL): "एभ्यः",
            (Case.GENITIVE, Number.PLURAL): "आनाम्",  # rāmāṇām
            (Case.LOCATIVE, Number.PLURAL): "एषु",  # rāmeṣu
            (Case.VOCATIVE, Number.PLURAL): "आः",
        }
    )
    
    # Neuter a-stems (like phala, dhana)
    NEUTER_A_STEM = DeclensionPattern(
        stem_class="a-stem",
        gender=Gender.NEUTER,
        endings={
            # Singular
            (Case.NOMINATIVE, Number.SINGULAR): "म्",  # phalam
            (Case.ACCUSATIVE, Number.SINGULAR): "म्",  # phalam
            (Case.INSTRUMENTAL, Number.SINGULAR): "एन",
            (Case.DATIVE, Number.SINGULAR): "आय",
            (Case.ABLATIVE, Number.SINGULAR): "आत्",
            (Case.GENITIVE, Number.SINGULAR): "अस्य",
            (Case.LOCATIVE, Number.SINGULAR): "ए",
            (Case.VOCATIVE, Number.SINGULAR): "म्",
            
            # Dual
            (Case.NOMINATIVE, Number.DUAL): "ए",  # phale
            (Case.ACCUSATIVE, Number.DUAL): "ए",
            (Case.INSTRUMENTAL, Number.DUAL): "आभ्याम्",
            (Case.DATIVE, Number.DUAL): "आभ्याम्",
            (Case.ABLATIVE, Number.DUAL): "आभ्याम्",
            (Case.GENITIVE, Number.DUAL): "अयोः",
            (Case.LOCATIVE, Number.DUAL): "अयोः",
            (Case.VOCATIVE, Number.DUAL): "ए",
            
            # Plural
            (Case.NOMINATIVE, Number.PLURAL): "आनि",  # phalāni
            (Case.ACCUSATIVE, Number.PLURAL): "आनि",
            (Case.INSTRUMENTAL, Number.PLURAL): "ऐः",
            (Case.DATIVE, Number.PLURAL): "एभ्यः",
            (Case.ABLATIVE, Number.PLURAL): "एभ्यः",
            (Case.GENITIVE, Number.PLURAL): "आनाम्",
            (Case.LOCATIVE, Number.PLURAL): "एषु",
            (Case.VOCATIVE, Number.PLURAL): "आनि",
        }
    )
    
    # Feminine ā-stems (like sītā, rāmā)
    FEMININE_A_STEM = DeclensionPattern(
        stem_class="ā-stem",
        gender=Gender.FEMININE,
        endings={
            # Singular
            (Case.NOMINATIVE, Number.SINGULAR): "आ",  # sītā
            (Case.ACCUSATIVE, Number.SINGULAR): "आम्",  # sītām
            (Case.INSTRUMENTAL, Number.SINGULAR): "अया",  # sītayā
            (Case.DATIVE, Number.SINGULAR): "आयै",  # sītāyai
            (Case.ABLATIVE, Number.SINGULAR): "आयाः",  # sītāyāḥ
            (Case.GENITIVE, Number.SINGULAR): "आयाः",
            (Case.LOCATIVE, Number.SINGULAR): "आयाम्",  # sītāyām
            (Case.VOCATIVE, Number.SINGULAR): "ए",  # sīte
            
            # Dual
            (Case.NOMINATIVE, Number.DUAL): "ए",  # sīte
            (Case.ACCUSATIVE, Number.DUAL): "ए",
            (Case.INSTRUMENTAL, Number.DUAL): "आभ्याम्",
            (Case.DATIVE, Number.DUAL): "आभ्याम्",
            (Case.ABLATIVE, Number.DUAL): "आभ्याम्",
            (Case.GENITIVE, Number.DUAL): "अयोः",
            (Case.LOCATIVE, Number.DUAL): "अयोः",
            (Case.VOCATIVE, Number.DUAL): "ए",
            
            # Plural
            (Case.NOMINATIVE, Number.PLURAL): "आः",  # sītāḥ
            (Case.ACCUSATIVE, Number.PLURAL): "आः",
            (Case.INSTRUMENTAL, Number.PLURAL): "आभिः",  # sītābhiḥ
            (Case.DATIVE, Number.PLURAL): "आभ्यः",  # sītābhyaḥ
            (Case.ABLATIVE, Number.PLURAL): "आभ्यः",
            (Case.GENITIVE, Number.PLURAL): "आनाम्",  # sītānām
            (Case.LOCATIVE, Number.PLURAL): "आसु",  # sītāsu
            (Case.VOCATIVE, Number.PLURAL): "आः",
        }
    )
    
    @classmethod
    def get_patterns(cls) -> List[DeclensionPattern]:
        """Get all declension patterns."""
        return [
            cls.MASCULINE_A_STEM,
            cls.NEUTER_A_STEM,
            cls.FEMININE_A_STEM
        ]
    
    @classmethod
    def get_pattern_by_ending(cls, ending: str) -> Optional[DeclensionPattern]:
        """Find declension pattern that has this ending."""
        for pattern in cls.get_patterns():
            if ending in pattern.endings.values():
                return pattern
        return None


@dataclass
class MorphologyPerturbation:
    """A controlled morphological perturbation."""
    phenomenon: str  # "case_swap", "number_swap", etc.
    from_case: Optional[Case] = None
    to_case: Optional[Case] = None
    from_number: Optional[Number] = None
    to_number: Optional[Number] = None
    from_ending: str = ""
    to_ending: str = ""
    
    def describe(self) -> str:
        """Human-readable description."""
        if self.phenomenon == "case_swap":
            return f"{self.from_case.value} → {self.to_case.value}"
        elif self.phenomenon == "number_swap":
            return f"{self.from_number.value} → {self.to_number.value}"
        else:
            return self.phenomenon


def generate_case_perturbations(
    pattern: DeclensionPattern,
    number: Number = Number.SINGULAR
) -> List[MorphologyPerturbation]:
    """Generate case perturbations for a given number.
    
    Args:
        pattern: Declension pattern
        number: Number to keep constant
        
    Returns:
        List of perturbations (swapping cases while keeping number constant)
    """
    perturbations = []
    
    cases = [Case.NOMINATIVE, Case.ACCUSATIVE, Case.INSTRUMENTAL,
             Case.DATIVE, Case.ABLATIVE, Case.GENITIVE, Case.LOCATIVE]
    
    for from_case in cases:
        for to_case in cases:
            if from_case != to_case:
                from_ending = pattern.endings.get((from_case, number))
                to_ending = pattern.endings.get((to_case, number))
                
                if from_ending and to_ending:
                    perturbations.append(MorphologyPerturbation(
                        phenomenon="case_swap",
                        from_case=from_case,
                        to_case=to_case,
                        from_number=number,
                        to_number=number,
                        from_ending=from_ending,
                        to_ending=to_ending
                    ))
    
    return perturbations


def generate_number_perturbations(
    pattern: DeclensionPattern,
    case: Case = Case.NOMINATIVE
) -> List[MorphologyPerturbation]:
    """Generate number perturbations for a given case.
    
    Args:
        pattern: Declension pattern
        case: Case to keep constant
        
    Returns:
        List of perturbations (swapping numbers while keeping case constant)
    """
    perturbations = []
    
    numbers = [Number.SINGULAR, Number.DUAL, Number.PLURAL]
    
    for from_number in numbers:
        for to_number in numbers:
            if from_number != to_number:
                from_ending = pattern.endings.get((case, from_number))
                to_ending = pattern.endings.get((case, to_number))
                
                if from_ending and to_ending:
                    perturbations.append(MorphologyPerturbation(
                        phenomenon="number_swap",
                        from_case=case,
                        to_case=case,
                        from_number=from_number,
                        to_number=to_number,
                        from_ending=from_ending,
                        to_ending=to_ending
                    ))
    
    return perturbations
