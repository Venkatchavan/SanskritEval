"""Generate mock probing results for testing/demonstration."""

import json
import numpy as np
from pathlib import Path


def generate_mock_results():
    """Generate realistic mock probing results.
    
    Simulates layer-wise patterns observed in real probing studies:
    - Early layers: Lower performance (surface features)
    - Middle layers: Peak performance (syntactic patterns)
    - Late layers: Slight decline (task-specific fine-tuning)
    """
    
    # mBERT has 12 layers + embedding layer = 13
    num_layers = 13
    
    # Sandhi boundaries: Peaks in middle layers (syntactic)
    sandhi_results = []
    for layer in range(num_layers):
        # Create realistic curve: peak around layer 6-8
        base_acc = 0.55 + 0.30 * np.exp(-((layer - 7) ** 2) / 8)
        noise = np.random.uniform(-0.02, 0.02)
        accuracy = min(0.95, max(0.50, base_acc + noise))
        
        # F1 slightly different from accuracy
        f1_score = accuracy + np.random.uniform(-0.03, 0.03)
        f1_score = min(0.95, max(0.50, f1_score))
        
        sandhi_results.append({
            'layer': layer,
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1_score, 4)
        })
    
    # Morphology: Peaks slightly later (semantic)
    morph_results = []
    for layer in range(num_layers):
        # Peak around layer 8-10
        base_acc = 0.58 + 0.32 * np.exp(-((layer - 9) ** 2) / 10)
        noise = np.random.uniform(-0.02, 0.02)
        accuracy = min(0.95, max(0.50, base_acc + noise))
        
        f1_score = accuracy + np.random.uniform(-0.03, 0.03)
        f1_score = min(0.95, max(0.50, f1_score))
        
        morph_results.append({
            'layer': layer,
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1_score, 4)
        })
    
    # Combine results
    results = [
        {
            'task': 'sandhi_boundaries',
            'model': 'bert-base-multilingual-cased',
            'num_layers': 12,
            'hidden_dim': 768,
            'results': sandhi_results
        },
        {
            'task': 'morphological_acceptability',
            'model': 'bert-base-multilingual-cased',
            'num_layers': 12,
            'hidden_dim': 768,
            'results': morph_results
        }
    ]
    
    return results


def main():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate results
    results = generate_mock_results()
    
    # Save to file
    output_path = Path('results/probing_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Mock probing results saved to {output_path}")
    print(f"\nGenerated {len(results)} experiments:")
    
    for result in results:
        task = result['task']
        best = max(result['results'], key=lambda x: x['f1_score'])
        print(f"\n{task}:")
        print(f"  Best layer: {best['layer']}")
        print(f"  Peak F1: {best['f1_score']:.3f}")
        print(f"  Peak accuracy: {best['accuracy']:.3f}")


if __name__ == '__main__':
    main()
