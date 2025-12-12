"""Transformer-based models for evaluation."""

from typing import List, Optional
import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base import AcceptabilityScorer

logger = logging.getLogger(__name__)


class MaskedLMScorer(AcceptabilityScorer):
    """Pseudo-likelihood scorer using masked language models.
    
    Computes sum of log P(token_i | context) by masking each token.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize masked LM scorer.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'bert-base-multilingual-cased')
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch required for MaskedLMScorer. "
                "Install with: pip install torch transformers"
            )
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def score(self, text: str) -> float:
        """Score text using pseudo-likelihood.
        
        Args:
            text: Text to score
            
        Returns:
            Sum of log probabilities (higher = more acceptable)
        """
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= 2:  # Only [CLS] and [SEP]
            return 0.0
        
        total_log_prob = 0.0
        
        # Mask each token and predict
        with torch.no_grad():
            for i in range(1, len(tokens) - 1):  # Skip [CLS] and [SEP]
                # Create masked input
                masked_tokens = tokens.copy()
                original_token = masked_tokens[i]
                masked_tokens[i] = self.tokenizer.mask_token_id
                
                # Get prediction
                input_ids = torch.tensor([masked_tokens]).to(self.device)
                outputs = self.model(input_ids)
                predictions = outputs.logits[0, i]
                
                # Get log probability of original token
                log_probs = torch.log_softmax(predictions, dim=0)
                token_log_prob = log_probs[original_token].item()
                total_log_prob += token_log_prob
        
        return total_log_prob
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """Score batch of texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of scores
        """
        return [self.score(text) for text in texts]


class ModelFactory:
    """Factory for creating model instances."""
    
    MODELS = {
        'mbert': 'bert-base-multilingual-cased',
        'xlm-r-base': 'xlm-roberta-base',
        'xlm-r-large': 'xlm-roberta-large',
        'indicbert': 'ai4bharat/indic-bert',
        'muril': 'google/muril-base-cased',
    }
    
    @classmethod
    def create_scorer(cls, model_key: str, device: Optional[str] = None) -> AcceptabilityScorer:
        """Create acceptability scorer.
        
        Args:
            model_key: Model identifier (key in MODELS dict or full HF name)
            device: Device to use
            
        Returns:
            AcceptabilityScorer instance
        """
        model_name = cls.MODELS.get(model_key, model_key)
        return MaskedLMScorer(model_name, device=device)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model keys.
        
        Returns:
            List of model identifiers
        """
        return list(cls.MODELS.keys())
