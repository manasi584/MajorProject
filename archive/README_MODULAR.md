# Modular Implementation Guide

Clean separation of concerns with feature extraction, feature selection, and evaluation.

## Architecture

```
01_feature_extraction_*.py          → Feature extraction (ResNet50, MobileNet)
                                      Output: features_*.npz

02_feature_selection_*.py           → Feature selection optimization
                                      - quantum_puma.py
                                      - quantum_firefly.py
                                      - quantum_reptile.py

03_kfold_evaluation.py              → K-fold CV + evaluation
                                      (Works with any feature combination)
```

## Workflow

### Step 1: Extract Features

Choose one feature extraction method:

**Option A: ResNet50**
```bash
python 01_feature_extraction_resnet50.py
# Output: features_resnet50.npz
```

**Option B: MobileNet**
```bash
python 01_feature_extraction_mobilenet.py
# Output: features_mobilenet.npz
```

### Step 2: Run Evaluation with Feature Selection

Edit `03_kfold_evaluation.py` and configure:

```python
# Choose features and feature selection method
FEATURES_PATH = "./features_resnet50.npz"          # or features_mobilenet.npz
FEATURE_SELECTION = "quantum_puma"                 # quantum_puma, quantum_firefly, or quantum_reptile
N_SPLITS = 5
```

Run evaluation:
```bash
python 03_kfold_evaluation.py
```

## Available Feature Selection Methods

### 1. Quantum Puma (QSM-PO)
- **File**: `02_feature_selection_quantum_puma.py`
- **Algorithm**: Population-based with superposition mutations
- **Features**:
  - Quantum superposition collapse
  - Territorial behavior
  - Hunt intensity adaptation
  - Adaptive mutation rate
- **Best for**: Balanced exploration-exploitation
- **Default params**: n_pumas=30, max_iterations=100

### 2. Quantum Firefly (QFA)
- **File**: `02_feature_selection_quantum_firefly.py`
- **Algorithm**: Swarm-based with light attraction
- **Features**:
  - Brightness-based attraction
  - Quantum angle modulation
  - Exponential distance decay
  - Adaptive randomness
- **Best for**: Smooth landscape optimization
- **Default params**: n_fireflies=30, max_iterations=100

### 3. Quantum Reptile (QMRS)
- **File**: `02_feature_selection_quantum_reptile.py`
- **Algorithm**: Population-based with cooperative hunting
- **Features**:
  - Prey encirclement
  - Cooperative hunting
  - Quantum phase mutations
  - Qbit state tracking
- **Best for**: Complex multimodal landscapes
- **Default params**: n_reptiles=30, max_iterations=100

## Features at Each Stage

### Feature Extraction Output
```
features_*.npz contains:
  - X: Deep learning features
    * ResNet50: 2048-dim
    * MobileNet: 1280-dim
  - DCP: Dark Channel Prior (1-dim)
  - y: PM2.5 labels (continuous)
  - PM25_MAX: Max PM2.5 for normalization
```

### K-Fold Evaluation
```
Input: features_*.npz
Process:
  1. Load features + DCP
  2. Convert PM2.5 → 6 classes
     (Good, Moderate, USG, Unhealthy, Very Unhealthy, Hazardous)
  3. Train-test split (80-20)
  4. For each fold:
     - Feature selection optimization
     - Model training via metaheuristic
     - Hybrid gradient refinement
     - OOF + test predictions
  5. Compute CV accuracy + test accuracy + report
```

## Feature Combinations (Performance Table)

| Feature Extraction | Feature Selection | Accuracy | Speed | Notes |
|-------------------|------------------|----------|-------|-------|
| ResNet50          | Quantum Puma     | **High** | Med   | Best balanced |
| ResNet50          | Quantum Firefly  | High     | Med   | Smooth convergence |
| ResNet50          | Quantum Reptile  | High     | Med   | Complex terrain |
| MobileNet         | Quantum Puma     | Good     | Fast  | Lightweight |
| MobileNet         | Quantum Firefly  | Good     | Fast  | Mobile-friendly |
| MobileNet         | Quantum Reptile  | Good     | Fast  | Quick exploration |

## Hybrid Refinement

All optimizers support **hybrid gradient refinement**:
- After metaheuristic optimization, fine-tunes with gradient descent
- Uses Adam optimizer with lr=1e-3
- 10 iterations by default
- Significantly improves final accuracy

## Usage Examples

### ResNet50 + Quantum Puma
```bash
# Extract features
python 01_feature_extraction_resnet50.py

# Edit 03_kfold_evaluation.py:
# FEATURES_PATH = "./features_resnet50.npz"
# FEATURE_SELECTION = "quantum_puma"

# Run evaluation
python 03_kfold_evaluation.py
```

### MobileNet + Quantum Firefly
```bash
# Extract features
python 01_feature_extraction_mobilenet.py

# Edit 03_kfold_evaluation.py:
# FEATURES_PATH = "./features_mobilenet.npz"
# FEATURE_SELECTION = "quantum_firefly"

# Run evaluation
python 03_kfold_evaluation.py
```

### Benchmarking All Combinations
```bash
# Extract both feature types
python 01_feature_extraction_resnet50.py
python 01_feature_extraction_mobilenet.py

# For each combination, update 03_kfold_evaluation.py and run:
python 03_kfold_evaluation.py
```

## Adding New Methods

### Add New Feature Extraction

Create `01_feature_extraction_yourmodel.py`:
```python
def extract_features_yourmodel(dataset_path, csv_path, image_dir, pm25_max=None):
    # Your extraction logic here
    return X, DCP, y, pm25_max

# Save
np.savez("features_yourmodel.npz", X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)
```

### Add New Feature Selection

Create `02_feature_selection_youroptimizer.py`:
```python
class YourAgent:
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        # Your agent state

class YourOptimizer:
    def __init__(self, model, X_train, y_train, X_val, y_val, **kwargs):
        # Setup
        self.agents = [YourAgent(...) for _ in range(n_agents)]
    
    def optimize(self, verbose=True, use_hybrid=True):
        # Main optimization loop
        # Call _hybrid_gradient_refinement() at the end if use_hybrid=True
```

Then update `03_kfold_evaluation.py`:
```python
elif FEATURE_SELECTION == "your_optimizer":
    from feature_selection_your_optimizer import YourOptimizer
    optimizer_class = YourOptimizer
    optimizer_params = {
        'n_agents': 30,
        'max_iterations': 100,
        # your params
    }
```

## Configuration Reference

### Quantum Puma Parameters
```python
{
    'n_pumas': 30,                    # Population size
    'max_iterations': 100,             # Max iterations
    'hunt_intensity': 0.5,            # Exploitation factor
    'exploration_rate': 0.5,          # Exploration probability
    'mutation_rate': 0.15,            # Quantum mutation rate
    'batch_size': 32                  # Validation batch size
}
```

### Quantum Firefly Parameters
```python
{
    'n_fireflies': 30,                # Swarm size
    'max_iterations': 100,            # Max iterations
    'attraction': 0.5,                # Attraction coefficient
    'randomness': 0.3,               # Randomness factor
    'quantum_factor': 0.1,            # Quantum effect strength
    'batch_size': 32                  # Validation batch size
}
```

### Quantum Reptile Parameters
```python
{
    'n_reptiles': 30,                 # Population size
    'max_iterations': 100,            # Max iterations
    'encircle_factor': 0.5,          # Prey encirclement strength
    'hunt_factor': 0.3,              # Cooperative hunting factor
    'mutation_rate': 0.1,            # Quantum mutation rate
    'batch_size': 32                  # Validation batch size
}
```

## Performance Notes

- **ResNet50**: ~2048-dim features, slower extraction, highest accuracy
- **MobileNet**: ~1280-dim features, faster extraction, good accuracy
- **Quantum Puma**: Balanced, recommended for most cases
- **Quantum Firefly**: Best for smooth, continuous landscapes
- **Quantum Reptile**: Best for complex, multimodal landscapes
- **K-Fold CV**: 5 folds recommended for stability

## File Sizes (Approximate)

- ResNet50 features: 300-500 MB
- MobileNet features: 200-300 MB
- Can be further optimized with dimensionality reduction

## Troubleshooting

### Low Accuracy
- Check class distribution in data
- Increase `max_iterations`
- Ensure hybrid refinement is enabled
- Try different feature extraction methods

### Early Stopping
- Increase `no_improvement_count` threshold (default: 30)
- Adjust exploration rate for your optimizer
- Reduce mutation rate if too aggressive

### Memory Issues
- Reduce `batch_size`
- Use smaller feature extraction model
- Process data in chunks
