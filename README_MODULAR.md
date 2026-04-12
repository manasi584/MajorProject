# Modular Implementation Guide

Clean separation of concerns with feature extraction, feature selection, and evaluation.

## Architecture

```
01_feature_extraction_*.py     → Feature extraction (ResNet50, MobileNet)
                                  Output: features_*.npz

02_feature_selection_*.py      → Feature selection optimization
                                  (Quantum Puma, Quantum Firefly, etc.)

03_kfold_evaluation.py         → K-fold CV + evaluation
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

Modify `03_kfold_evaluation.py`:

```python
# Choose features and feature selection method
FEATURES_PATH = "./features_resnet50.npz"      # or features_mobilenet.npz
FEATURE_SELECTION = "quantum_puma"              # quantum_puma, quantum_firefly, etc.
N_SPLITS = 5
```

Run evaluation:
```bash
python 03_kfold_evaluation.py
```

## Features at Each Stage

### Feature Extraction Output
```
features_*.npz contains:
  - X: Deep learning features (ResNet50: 2048-dim, MobileNet: 1280-dim)
  - DCP: Dark Channel Prior (1-dim)
  - y: PM2.5 labels (continuous)
  - PM25_MAX: Max PM2.5 for normalization
```

### K-Fold Evaluation
```
Input: features_*.npz
Process:
  1. Load features + DCP
  2. Convert PM2.5 → 6 classes (Good, Moderate, USG, Unhealthy, Very Unhealthy, Hazardous)
  3. Train-test split (80-20)
  4. For each fold:
     - Feature selection optimization (e.g., Quantum Puma)
     - Model training via evolution
     - OOF + test predictions
  5. Compute CV accuracy + test accuracy + classification report
```

## Feature Combinations

You can mix and match:

| Feature Extraction | Feature Selection | Expected Performance |
|-------------------|------------------|----------------------|
| ResNet50          | Quantum Puma     | High accuracy        |
| ResNet50          | Quantum Firefly  | High accuracy        |
| MobileNet         | Quantum Puma     | Faster + good        |
| MobileNet         | Quantum Firefly  | Faster + good        |

## Adding New Methods

### Add New Feature Extraction

Create `01_feature_extraction_yourmodel.py`:
```python
def extract_features_yourmodel(dataset_path, csv_path, image_dir, pm25_max=None):
    # Your extraction logic
    return X, DCP, y, pm25_max

# Save
np.savez("features_yourmodel.npz", X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)
```

### Add New Feature Selection

Create `02_feature_selection_youroptimizer.py`:
```python
class YourOptimizer:
    def __init__(self, model, X_train, y_train, X_val, y_val, **kwargs):
        # Setup
    
    def optimize(self, verbose=True):
        # Optimization logic
```

Then update `03_kfold_evaluation.py`:
```python
elif FEATURE_SELECTION == "your_optimizer":
    from feature_selection_your_optimizer import YourOptimizer
    optimizer_class = YourOptimizer
    optimizer_params = {
        # your params
    }
```

## Example Runs

### ResNet50 + Quantum Puma
```bash
python 01_feature_extraction_resnet50.py
# Then in 03_kfold_evaluation.py:
FEATURES_PATH = "./features_resnet50.npz"
FEATURE_SELECTION = "quantum_puma"
python 03_kfold_evaluation.py
```

### MobileNet + Quantum Puma
```bash
python 01_feature_extraction_mobilenet.py
# Then in 03_kfold_evaluation.py:
FEATURES_PATH = "./features_mobilenet.npz"
FEATURE_SELECTION = "quantum_puma"
python 03_kfold_evaluation.py
```

## Performance Notes

- **ResNet50**: ~2048-dim features, slower extraction, highest accuracy
- **MobileNet**: ~1280-dim features, faster extraction, good accuracy
- **Quantum Puma**: Population-based optimization, good exploration
- **K-Fold CV**: 5 folds recommended for stability

## File Sizes (Approximate)

- ResNet50 features: 300-500 MB
- MobileNet features: 200-300 MB
- Can be further compressed with feature selection
