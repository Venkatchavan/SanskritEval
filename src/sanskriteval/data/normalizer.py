"""Data normalization module for Sanskrit text processing."""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict

try:
    import jsonlines
    JSONLINES_AVAILABLE = True
except ImportError:
    JSONLINES_AVAILABLE = False

from ..utils.text_processing import (
    clean_sanskrit_text,
    convert_script,
    extract_verse_id,
    ScriptType
)
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class NormalizedVerse:
    """Normalized verse data structure."""
    id: str
    text: str
    source: str
    script: ScriptType
    chapter: Optional[int] = None
    verse: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class DataNormalizer:
    """Normalize Sanskrit text data with consistent representation."""
    
    def __init__(
        self,
        output_script: ScriptType = "iast",
        source_name: str = "bhagavad_gita"
    ):
        """Initialize normalizer.
        
        Args:
            output_script: Target script for output (devanagari, iast, slp1)
            source_name: Name of the source text
        """
        self.output_script = output_script
        self.source_name = source_name
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
    
    def normalize_verse(
        self,
        text: str,
        verse_id: Optional[str] = None,
        input_script: ScriptType = "devanagari"
    ) -> NormalizedVerse:
        """Normalize a single verse.
        
        Args:
            text: Raw verse text
            verse_id: Verse identifier (e.g., "1.1")
            input_script: Script of input text
            
        Returns:
            NormalizedVerse object
        """
        # Extract verse ID if not provided
        if verse_id is None:
            verse_id = extract_verse_id(text)
            if verse_id is None:
                raise ValueError("Could not extract verse ID from text")
        
        # Clean text
        cleaned_text = clean_sanskrit_text(text)
        
        # Convert script if needed
        if input_script != self.output_script:
            cleaned_text = convert_script(
                cleaned_text,
                source_script=input_script,
                target_script=self.output_script
            )
        
        # Parse chapter and verse numbers
        chapter, verse = None, None
        if '.' in verse_id:
            parts = verse_id.split('.')
            chapter = int(parts[0])
            verse = int(parts[1])
        
        return NormalizedVerse(
            id=verse_id,
            text=cleaned_text,
            source=self.source_name,
            script=self.output_script,
            chapter=chapter,
            verse=verse
        )
    
    def normalize_file(
        self,
        input_path: Path,
        output_path: Path,
        input_format: Literal["txt", "json", "jsonl"] = "txt",
        input_script: ScriptType = "devanagari"
    ) -> int:
        """Normalize an entire file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output JSONL file
            input_format: Format of input file
            input_script: Script of input text
            
        Returns:
            Number of verses processed
        """
        self.logger.info(f"Normalizing {input_path} -> {output_path}")
        
        verses = []
        
        if input_format == "txt":
            verses = self._read_txt(input_path, input_script)
        elif input_format == "json":
            verses = self._read_json(input_path, input_script)
        elif input_format == "jsonl":
            verses = self._read_jsonl(input_path, input_script)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        # Write normalized output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not JSONLINES_AVAILABLE:
            raise ImportError("jsonlines package required. Install with: pip install jsonlines")
        
        with jsonlines.open(output_path, mode='w') as writer:
            for verse in verses:
                writer.write(verse.to_dict())
        
        self.logger.info(f"Wrote {len(verses)} verses to {output_path}")
        return len(verses)
    
    def _read_txt(
        self,
        path: Path,
        input_script: ScriptType
    ) -> List[NormalizedVerse]:
        """Read plain text file (one verse per line with ID)."""
        verses = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try to extract verse ID from line
                    verse_id = extract_verse_id(line)
                    if verse_id:
                        # Remove ID from text
                        text = re.sub(r'\d+\.\d+\s*[:\-]?\s*', '', line)
                    else:
                        # Use line number as fallback
                        verse_id = str(line_num)
                        text = line
                    
                    verse = self.normalize_verse(text, verse_id, input_script)
                    verses.append(verse)
                except Exception as e:
                    self.logger.warning(f"Failed to process line {line_num}: {e}")
        
        return verses
    
    def _read_json(
        self,
        path: Path,
        input_script: ScriptType
    ) -> List[NormalizedVerse]:
        """Read JSON file with verses."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        verses = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'verses' in data:
            items = data['verses']
        else:
            raise ValueError("JSON must be list or dict with 'verses' key")
        
        for item in items:
            verse = self.normalize_verse(
                text=item.get('text', ''),
                verse_id=item.get('id') or item.get('verse_id'),
                input_script=input_script
            )
            verses.append(verse)
        
        return verses
    
    def _read_jsonl(
        self,
        path: Path,
        input_script: ScriptType
    ) -> List[NormalizedVerse]:
        """Read JSONL file."""
        if not JSONLINES_AVAILABLE:
            raise ImportError("jsonlines package required. Install with: pip install jsonlines")
            
        verses = []
        with jsonlines.open(path) as reader:
            for item in reader:
                verse = self.normalize_verse(
                    text=item.get('text', ''),
                    verse_id=item.get('id') or item.get('verse_id'),
                    input_script=input_script
                )
                verses.append(verse)
        
        return verses


# End of module
