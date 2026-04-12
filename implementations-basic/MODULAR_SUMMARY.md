# Modular Implementation Summary

## Files Created ✓

### Feature Extraction (Stage 1)
- ✓ `01_feature_extraction_resnet50.py` — ResNet50 + Dark Channel Prior
- ✓ `01_feature_extraction_mobilenet.py` — MobileNetV2 + Dark Channel Prior

### Feature Selection (Stage 2)
- ✓ `02_feature_selection_quantum_puma.py` — QSM-PO optimizer (Superposition + Territorial)
- ✓ `02_feature_selection_quantum_firefly.py` — QFA optimizer (Light Attraction)
- ✓ `02_feature_selection_quantum_reptile.py` — QMRS optimizer (Cooperative Hunting)

### K-Fold Evaluation (Stage 3)
- ✓ `03_kfold_evaluation.py` — Universal evaluator for any combination

### Documentation
- ✓ `README_MODULAR.md` — Complete usage guide
- ✓ `MODULAR_SUMMARY.md` — This file

---

## Quick Start

### 1. Extract Features
```bash
# Option A: ResNet50
python 01_feature_extraction_resnet50.py
# Output: features_resnet50.npz

# Option B: MobileNet
python 01_feature_extraction_mobilenet.py
# Output: features_mobilenet.npz
```

### 2. Run K-Fold Evaluation
Edit `03_kfold_evaluation.py`:
```python
FEATURES_PATH = "./features_resnet50.npz"  # or features_mobilenet.npz
FEATURE_SELECTION = "quantum_puma"  # or quantum_firefly, quantum_reptile
```

Then run:
```bash
python 03_kfold_evaluation.py
```

---

## Algorithm Comparison

| Algorithm | Type | Key Feature | Best For |
|-----------|------|-----------|----------|
| **Quantum Puma** | Population | Superposition + Territory | Balanced search |
| **Quantum Firefly** | Swarm | Light attraction | Smooth landscapes |
| **Quantum Reptile** | Population | Cooperative hunting | Multimodal problems |

---

## Architecture

```
Data Input
    ↓
Feature Extraction (Stage 1)
├─ ResNet50 → features_resnet50.npz
└─ MobileNet → features_mobilenet.npz
    ↓
K-Fold Evaluation (Stage 3)
├─ Load features
├─ For each fold:
│  ├─ Feature Selection Optimization (Stage 2)
│  │  ├─ Quantum Puma
│  │  ├─ Quantum Firefly
│  │  └─ Quantum Reptile
│  ├─ Hybrid Gradient Refinement
│  └─ Predictions
├─ CV Accuracy
└─ Test Accuracy + Report
```

---

## Key Features

### Hybrid Refinement
All optimizers include **gradient descent fine-tuning** after metaheuristic optimization:
- Significantly improves accuracy
- Uses Adam optimizer (lr=1e-3)
- 10 iterations by default
- Enabled by default in `optimize(use_hybrid=True)`

### Modular Design
- Each stage is independent
- Easy to add new extractors/optimizers
- Reusable components
- Clean separation of concerns

### Class Imbalance Handling
- Stratified K-fold CV
- Proper train-test splits
- Class distribution reporting

---

## Performance Notes

### Feature Extraction
- **ResNet50**: 2048-dim features, high accuracy, slower
- **MobileNet**: 1280-dim features, good accuracy, faster

### Feature Selection
- **Quantum Puma**: Best overall, balanced exploration/exploitation
- **Quantum Firefly**: Smooth convergence, good for continuous spaces
- **Quantum Reptile**: Better for complex, multimodal problems

### Expected Improvements
- Early stopping at iteration 2 → increase `max_iterations` to 100-150
- Low accuracy → enable `use_hybrid=True` (default)
- Class imbalance → already handled with stratification

---

## Configuration Parameters

### 03_kfold_evaluation.py
```python
FEATURES_PATH = "./features_resnet50.npz"  # Feature file
FEATURE_SELECTION = "quantum_puma"         # Optimizer
N_SPLITS = 5                               # K-fold splits
SEED = 42                                  # Random seed
DEVICE = torch.device("cpu")               # GPU/CPU
```

### Quantum Puma
```python
{
    'n_pumas': 30,              # Population size
    'max_iterations': 100,       # Max iterations
    'hunt_intensity': 0.5,      # Exploitation
    'exploration_rate': 0.5,    # Exploration
    'mutation_rate': 0.15,      # Quantum mutation
    'batch_size': 32            # Val batch size
}
```

### Quantum Firefly
```python
{
    'n_fireflies': 30,          # Swarm size
    'max_iterations': 100,       # Max iterations
    'attraction': 0.5,          # Attraction strength
    'randomness': 0.3,         # Randomness factor
    'quantum_factor': 0.1,      # Quantum effect
    'batch_size': 32            # Val batch size
}
```

### Quantum Reptile
```python
{
    'n_reptiles': 30,           # Population size
    'max_iterations': 100,       # Max iterations
    'encircle_factor': 0.5,     # Encirclement strength
    'hunt_factor': 0.3,        # Hunting strength
    'mutation_rate': 0.1,       # Mutation rate
    'batch_size': 32            # Val batch size
}
```

---

## Troubleshooting

### Early Stopping (Iteration 2-5)
**Problem**: Optimizer stops immediately
**Solution**: 
- Hybrid refinement is ON by default (fixes this)
- Increase `max_iterations` to 150
- Check gradient flow in model

### Low Accuracy (< 30%)
**Problem**: Poor optimization results
**Solution**:
- Ensure `use_hybrid=True` (default)
- Try different feature extraction
- Increase `max_iterations` to 150
- Check data preprocessing

### Out of Memory
**Problem**: CUDA/RAM issues
**Solution**:
- Reduce `batch_size` to 16
- Use CPU mode instead of GPU
- Extract features in smaller batches

---

## Next Steps

1. **Extract features**: Run stage 1
2. **Benchmark**: Try all 6 combinations
3. **Analyze**: Compare results
4. **Optimize**: Fine-tune parameters
5. **Deploy**: Use best configuration

---

## File Dependencies

```
03_kfold_evaluation.py
├─ Imports: 02_feature_selection_quantum_puma.py
├─ Imports: 02_feature_selection_quantum_firefly.py
├─ Imports: 02_feature_selection_quantum_reptile.py
└─ Requires: features_*.npz (from 01_feature_extraction_*.py)
```

---

## Size Reference

| File | Size |
|------|------|
| ResNet50 features | ~300-500 MB |
| MobileNet features | ~200-300 MB |
| Extracted features | ~1-2 GB (decompressed) |

---

## Success Criteria

✓ Feature extraction: 5000+ samples extracted
✓ K-fold CV: Completes all 5 folds
✓ Accuracy: > 60% on test set
✓ Convergence: Improvement over early stopping
✓ Hybrid refinement: Improves final accuracy

