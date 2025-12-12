"""Layer-wise probing for interpretability analysis."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

try:
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Results from probing a single layer."""
    layer_idx: int
    accuracy: float
    f1_score: float
    num_train: int
    num_test: int


class LinearProbe:
    """Linear probe for layer-wise analysis.
    
    Trains a logistic regression classifier on frozen hidden states
    to predict task-relevant properties.
    """
    
    def __init__(self, input_dim: int, random_state: int = 42):
        """Initialize probe.
        
        Args:
            input_dim: Dimension of hidden states
            random_state: Random seed for reproducibility
        """
        self.probe = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver='lbfgs'
        )
        self.input_dim = input_dim
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train probe on hidden states.
        
        Args:
            X: Hidden states (n_samples, hidden_dim)
            y: Binary labels (n_samples,)
        """
        self.probe.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels from hidden states.
        
        Args:
            X: Hidden states (n_samples, hidden_dim)
            
        Returns:
            Predicted labels (n_samples,)
        """
        return self.probe.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Compute accuracy and F1 score.
        
        Args:
            X: Hidden states
            y: True labels
            
        Returns:
            (accuracy, f1_score)
        """
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='binary')
        return acc, f1


class LayerWiseProber:
    """Extract hidden states from all layers and train probes."""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        """Initialize prober.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "Required: torch, transformers, sklearn. "
                "Install with: pip install torch transformers scikit-learn"
            )
        
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.model.to(device)
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.hidden_dim} hidden dim")
    
    def extract_hidden_states(
        self,
        texts: List[str],
        token_indices: List[int] = None
    ) -> Dict[int, np.ndarray]:
        """Extract hidden states from all layers.
        
        Args:
            texts: List of input texts
            token_indices: Specific token positions to extract (None = use [CLS])
            
        Returns:
            Dictionary mapping layer_idx -> hidden_states array
        """
        all_hidden_states = {i: [] for i in range(self.num_layers + 1)}  # +1 for embeddings
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
                
                # Extract [CLS] token representation from each layer
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    # Use [CLS] token (position 0)
                    cls_hidden = layer_hidden[0, 0, :].cpu().numpy()
                    all_hidden_states[layer_idx].append(cls_hidden)
        
        # Convert lists to arrays
        return {
            layer_idx: np.array(states)
            for layer_idx, states in all_hidden_states.items()
        }
    
    def probe_all_layers(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        test_texts: List[str],
        test_labels: np.ndarray,
        random_state: int = 42
    ) -> List[ProbeResult]:
        """Train probes on all layers and evaluate.
        
        Args:
            train_texts: Training texts
            train_labels: Training binary labels
            test_texts: Test texts
            test_labels: Test binary labels
            random_state: Random seed
            
        Returns:
            List of ProbeResult for each layer
        """
        logger.info("Extracting hidden states from training data...")
        train_hidden = self.extract_hidden_states(train_texts)
        
        logger.info("Extracting hidden states from test data...")
        test_hidden = self.extract_hidden_states(test_texts)
        
        results = []
        
        for layer_idx in range(self.num_layers + 1):
            logger.info(f"Probing layer {layer_idx}/{self.num_layers}...")
            
            # Get hidden states for this layer
            X_train = train_hidden[layer_idx]
            X_test = test_hidden[layer_idx]
            
            # Train probe
            probe = LinearProbe(self.hidden_dim, random_state=random_state)
            probe.train(X_train, train_labels)
            
            # Evaluate
            acc, f1 = probe.score(X_test, test_labels)
            
            result = ProbeResult(
                layer_idx=layer_idx,
                accuracy=acc,
                f1_score=f1,
                num_train=len(train_labels),
                num_test=len(test_labels)
            )
            results.append(result)
            
            logger.info(f"  Layer {layer_idx}: Acc={acc:.3f}, F1={f1:.3f}")
        
        return results


def prepare_sandhi_probe_data(
    examples: List[Tuple[str, str, bool]],
    train_ratio: float = 0.7
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Prepare data for sandhi boundary probing.
    
    Args:
        examples: List of (text, segmented, has_boundaries) tuples
        train_ratio: Ratio for train/test split
        
    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    texts = [ex[0] for ex in examples]
    labels = np.array([1 if ex[2] else 0 for ex in examples])
    
    # Split
    n_train = int(len(examples) * train_ratio)
    
    train_texts = texts[:n_train]
    train_labels = labels[:n_train]
    test_texts = texts[n_train:]
    test_labels = labels[n_train:]
    
    return train_texts, train_labels, test_texts, test_labels


def prepare_morphology_probe_data(
    pairs: List[Tuple[str, str]],
    train_ratio: float = 0.7
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Prepare data for morphology acceptability probing.
    
    Args:
        pairs: List of (grammatical, ungrammatical) tuples
        train_ratio: Ratio for train/test split
        
    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    # Flatten pairs into examples with labels
    texts = []
    labels = []
    
    for gram, ungram in pairs:
        texts.append(gram)
        labels.append(1)  # grammatical
        texts.append(ungram)
        labels.append(0)  # ungrammatical
    
    labels = np.array(labels)
    
    # Split
    n_train = int(len(texts) * train_ratio)
    
    train_texts = texts[:n_train]
    train_labels = labels[:n_train]
    test_texts = texts[n_train:]
    test_labels = labels[n_train:]
    
    return train_texts, train_labels, test_texts, test_labels
