"""Text processing utilities for Sanskrit text normalization."""

import re
import unicodedata
from typing import Literal, Optional

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False


ScriptType = Literal["devanagari", "iast", "slp1"]


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text to consistent form.
    
    Args:
        text: Input text
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize(form, text)


def remove_zero_width_chars(text: str) -> str:
    """Remove zero-width characters and other invisible Unicode.
    
    Args:
        text: Input text
        
    Returns:
        Text without zero-width characters
    """
    # Zero-width characters to remove
    zero_width = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space (BOM)
    ]
    
    for char in zero_width:
        text = text.replace(char, '')
    
    return text


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation variants to standard forms.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized punctuation
    """
    replacements = {
        # Various danda forms to standard danda
        '\u0964': '।',  # Devanagari danda
        '\u0965': '॥',  # Devanagari double danda
        
        # Spaces
        '\u00a0': ' ',  # Non-breaking space to regular space
        '\u2003': ' ',  # Em space
        '\u2002': ' ',  # En space
        
        # Quotes (normalize to standard)
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def convert_script(
    text: str,
    source_script: ScriptType = "devanagari",
    target_script: ScriptType = "iast"
) -> str:
    """Convert text between different Sanskrit scripts.
    
    Args:
        text: Input text
        source_script: Source script type
        target_script: Target script type
        
    Returns:
        Converted text
    """
    if not TRANSLITERATION_AVAILABLE:
        raise ImportError(
            "indic-transliteration package required for script conversion. "
            "Install with: pip install indic-transliteration"
        )
    
    script_map = {
        "devanagari": sanscript.DEVANAGARI,
        "iast": sanscript.IAST,
        "slp1": sanscript.SLP1
    }
    
    if source_script not in script_map or target_script not in script_map:
        raise ValueError(f"Unsupported script. Use: {list(script_map.keys())}")
    
    return transliterate(
        text,
        script_map[source_script],
        script_map[target_script]
    )


def clean_sanskrit_text(
    text: str,
    normalize_unicode_form: bool = True,
    remove_zero_width: bool = True,
    normalize_punct: bool = True
) -> str:
    """Apply all text cleaning operations.
    
    Args:
        text: Input text
        normalize_unicode_form: Apply Unicode normalization
        remove_zero_width: Remove zero-width characters
        normalize_punct: Normalize punctuation
        
    Returns:
        Cleaned text
    """
    if normalize_unicode_form:
        text = normalize_unicode(text)
    
    if remove_zero_width:
        text = remove_zero_width_chars(text)
    
    if normalize_punct:
        text = normalize_punctuation(text)
    
    return text


def extract_verse_id(text: str, pattern: Optional[str] = None) -> Optional[str]:
    """Extract verse ID from text using pattern.
    
    Args:
        text: Input text
        pattern: Regex pattern for verse ID (default: chapter.verse format)
        
    Returns:
        Extracted verse ID or None
    """
    if pattern is None:
        # Default pattern for Bhagavad Gita: "1.1", "2.15", etc.
        pattern = r'(\d+\.\d+)'
    
    match = re.search(pattern, text)
    return match.group(1) if match else None
