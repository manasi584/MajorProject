# Quantum Firefly Algorithm Pipeline - Complete Flowchart

## High-Level Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTUM FIREFLY PIPELINE                       │
└─────────────────────────────────────────────────────────────────────┘

         ┌──────────────────────────────┐
         │   Load Feature Extraction    │
         │  (ResNet50/MobileNet/EfficientNet) │
         │   - X features shape[samples, features] │
         │   - DCP metadata             │
         │   - y labels (normalized PM2.5) │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   Combine Features & DCP     │
         │  X_combined = [X | DCP]      │
         │  Shape: [samples, dim]       │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────────────┐
         │  Train-Val Split for Feature Sel.   │
         │  (80% train, 20% val)               │
         │  Stratified by PM2.5 classes        │
         └──────────────┬───────────────────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
              ▼                    ▼
    ┌──────────────────┐  ┌────────────────────┐
    │ X_train (train)  │  │ X_val (validate)   │
    │ y_train (labels) │  │ y_val (labels)     │
    └──────┬───────────┘  └──────┬─────────────┘
           │                     │
           └─────────────────────┴──────────────┐
                                                │
                                  ┌─────────────▼──────────────┐
                                  │   QUANTUM FIREFLY PHASE   │
                                  │    (Feature Selection)    │
                                  └──────────────┬────────────┘
                                                 │
         ┌───────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────┐
    │  Create Feature Selector Model          │
    │  - Input: n_features                    │
    │  - Hidden: 64 neurons (ReLU)           │
    │  - Output: 6 classes (PM2.5 AQI)       │
    └────────────┬────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Initialize Quantum Firefly Population       │
    │  - n_fireflies: 30 agents                    │
    │  - Position: random model parameters         │
    │  - Brightness: loss value (lower = better)  │
    │  - Quantum states: probability amplitudes   │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  MAIN OPTIMIZATION LOOP                      │
    │  (max_iterations: 100)                       │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ╔══════════════════════════════════════════════╗
    ║  ITERATION START                             ║
    ║  Loop: for each iteration (i=1 to 100)      ║
    ╚════────────────┬─────────────────────────────╝
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │  Sort Fireflies by Brightness               │
    │  (ascending order - best first)              │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  FOR EACH FIREFLY (from least to best)       │
    └────────────┬─────────────────────────────────┘
                 │
      ┌──────────┴─────────────┐
      │                        │
      ▼                        ▼
    ┌──────────────────────┐  ┌────────────────────┐
    │ Attraction Phase:    │  │ Quantum Phase:     │
    │                      │  │                    │
    │ For each brighter    │  │ Apply quantum      │
    │ firefly (j < i):     │  │ superposition      │
    │                      │  │ - Entanglement     │
    │ 1. Calculate dist    │  │ - Interference     │
    │ 2. Beta = exp(-γ·d²) │  │ - Collapse to      │
    │ 3. Force = β·(j-i)   │  │   classical state  │
    │ 4. Adaptive params   │  │                    │
    │    - attraction      │  │ Position update:   │
    │    - randomness      │  │ p' = p + Q·F + R   │
    │ 5. Update position   │  │ where Q is quantum │
    │    p' = p+F+Random   │  │ factor (0.1)       │
    │                      │  │                    │
    │ 6. Clip to bounds    │  └────────────────────┘
    │    [-0.1, 0.1]       │
    └────────┬─────────────┘
             │
             ▼
    ┌──────────────────────────────────────────────┐
    │  Evaluate Brightness:                        │
    │  - Assign position params to model           │
    │  - Forward pass on X_val                     │
    │  - Compute loss (CrossEntropyLoss)          │
    │  - Brightness = loss value                   │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Update Personal Best                        │
    │  if loss < firefly.best_brightness:          │
    │    save position as personal best            │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Update Global Best                          │
    │  if loss < global_best_brightness:           │
    │    save as global best                       │
    │    reset no_improvement_count                │
    │  else:                                       │
    │    increment no_improvement_count            │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Check Early Stopping Criterion              │
    │  if no_improvement_count > 30:               │
    │    BREAK from optimization loop              │
    │  else:                                       │
    │    continue to next iteration                │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Log Progress (every 20 iterations)          │
    │  Print: iteration, best_loss                 │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ╔══════════════════════════════════════════════╗
    ║  ITERATION END                               ║
    ║  Continue or break based on criteria        ║
    ╚════────────────┬─────────────────────────────╝
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │  END MAIN LOOP / OPTIMIZATION COMPLETE       │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Apply Global Best Weights to Model          │
    │  Set model.parameters = global_best_position │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  HYBRID REFINEMENT (Optional)                │
    │  Fine-tune with Gradient Descent             │
    │                                              │
    │  - Use Adam optimizer (lr=1e-3)              │
    │  - Train on X_train for 10 iterations       │
    │  - Each iteration:                           │
    │    * Forward pass, backward pass             │
    │    * Gradient updates                        │
    │    * Validate and update best if improved   │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Extract Feature Importance                  │
    │  - Get input layer weights (fc1.weight)      │
    │  - Sum absolute values per feature           │
    │  - Rank by importance magnitude              │
    │  - Select top N features (500)               │
    │  - Sort indices                              │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Return Selected Feature Indices             │
    │  Format: sorted array of indices             │
    └────────────┬─────────────────────────────────┘
                 │
         ┌───────┴────────────────────────┐
         │                                │
         ▼                                ▼
    ┌─────────────────┐          ┌────────────────────────┐
    │ X_selected =    │          │  K-FOLD CV PHASE      │
    │ X[:, indices]   │          │  (Model Evaluation)   │
    │ Reduced dim     │          └──────────┬─────────────┘
    └────────┬────────┘                     │
             │                              │
             └──────────────────────────────┘
                      │
                      ▼
    ┌──────────────────────────────────────────────┐
    │  Train-Test Split (80-20)                    │
    │  X_train_full (80%), X_test (20%)           │
    │  Stratified by class                         │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ╔══════════════════════════════════════════════╗
    ║  K-FOLD CROSS-VALIDATION LOOP                ║
    ║  for fold in range(5):                       ║
    ╚════────────────┬─────────────────────────────╝
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │  Split Train Data for Fold                   │
    │  - X_train, X_val from X_train_full         │
    │  - y_train, y_val from y_train_full         │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Build Simple FC Model                       │
    │  - Input: selected feature dim               │
    │  - Dense(256, ReLU) -> Dropout(0.3)         │
    │  - Dense(128, ReLU) -> Dropout(0.3)         │
    │  - Dense(6, softmax)                         │
    │  Optimizer: Adam                             │
    │  Loss: sparse_categorical_crossentropy       │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Train Model                                 │
    │  - Epochs: 30 (with early stopping)         │
    │  - Batch size: 32                            │
    │  - Callbacks:                                │
    │    * ModelCheckpoint (best val accuracy)    │
    │    * EarlyStopping (patience=10)             │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Generate OOF & Test Predictions             │
    │  - OOF: predictions on validation fold       │
    │  - Test: average predictions across folds    │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  Repeat for Next Fold                        │
    │  (if fold < n_splits)                        │
    └────────────┬─────────────────────────────────┘
                 │
    ╚════════════════╤═════════════════════════════╝
                     │
                     ▼
    ┌──────────────────────────────────────────────┐
    │  COMPUTE FINAL METRICS                       │
    │                                              │
    │  CV Accuracy:                                │
    │  - Aggregate OOF predictions                 │
    │  - Compare with training labels              │
    │                                              │
    │  Test Accuracy:                              │
    │  - Aggregate test predictions                │
    │  - Compare with test labels                  │
    │                                              │
    │  Classification Report:                      │
    │  - Precision, Recall, F1-score              │
    │  - Per-class and macro averages             │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  SAVE RESULTS                                │
    │                                              │
    │  Output file format:                         │
    │  {feature_name}_{method}.txt                │
    │                                              │
    │  Contains:                                   │
    │  - Configuration details                     │
    │  - Optimization progress                     │
    │  - Final accuracies (CV & Test)             │
    │  - Classification report                     │
    └────────────┬─────────────────────────────────┘
                 │
                 ▼
    ┌──────────────────────────────────────────────┐
    │  PIPELINE COMPLETE ✓                         │
    │                                              │
    │  Output:                                     │
    │  - Selected feature indices                  │
    │  - CV accuracy score                         │
    │  - Test accuracy score                       │
    │  - Model checkpoints for each fold           │
    │  - Detailed results log                      │
    └──────────────────────────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. **Data Loading & Preparation**
```
Load Features
├── X: CNN extracted features (ResNet50/MobileNet/EfficientNet)
├── DCP: Metadata features
├── y: Normalized PM2.5 values (0-1)
└── PM25_MAX: Denormalization factor
```

### 2. **Quantum Firefly Algorithm Components**

#### A. Firefly Initialization
```
Each Firefly Agent:
├── position: Random parameters in [-0.1, 0.1]
├── brightness: Loss value (lower = better)
├── best_position: Personal best found
└── best_brightness: Personal best loss
```

#### B. Quantum Superposition Phase
```
For each firefly position update:
├── Classical Attraction: p' = p + β(p_best - p)
├── Quantum Amplitudes: |α|² represents probability
├── Interference Effects: Phase cancellation/reinforcement
└── Collapse to Classical: Draw from probability distribution
```

#### C. Adaptive Parameters
```
Iteration-based decay:
├── attraction(t) = attraction₀ × (1 - t/T)
├── randomness(t) = randomness₀ × (1 - t/2T)
└── quantum_factor = 0.1 (constant)
where t = current iteration, T = max iterations
```

### 3. **Feature Selection through Weight Analysis**
```
Input Layer Weights Extraction:
├── Model fc1.weight shape: [64, n_features]
├── Absolute values: |w|
├── Sum across neurons: importance[j] = Σ|w[i,j]|
├── Sort and rank features by importance
└── Select top N features (500)
```

### 4. **K-Fold Evaluation**
```
For each fold (5 splits):
├── Train: 80% of data (minus validation fold)
├── Validation: 20% of training data
├── Test: 20% held out from start
├── Model: 256 → 128 → 6 (FC layers)
├── Early Stopping: patience=10
└── Metrics: Accuracy, Precision, Recall, F1-Score
```

---

## Key Decision Points in the Algorithm

### Early Stopping Criteria
- **Condition**: `no_improvement_count > 30`
- **Impact**: Prevents wasteful iterations when optimization plateaus
- **Benefit**: Reduces computation time while maintaining quality

### Hybrid Refinement
- **When**: After swarm optimization completes
- **How**: 10 iterations of gradient descent (Adam optimizer)
- **Purpose**: Fine-tune the best solution found by the swarm

### Feature Selection Threshold
- **Method**: Top-K selection on weight magnitudes
- **N Features**: 500 (configurable)
- **Rationale**: Balance between dimensionality reduction and information retention

---

## Quantum Enhancements Over Classical Firefly

| Aspect | Classical | Quantum |
|--------|-----------|---------|
| **State Representation** | Single position | Superposition of amplitudes |
| **Exploration** | Random walk | Quantum tunneling via interference |
| **Convergence** | Local optima prone | Better escaping local minima |
| **Computational Cost** | Low | Higher (amplitude tracking) |
| **Parameter Space** | 1 position/firefly | Position + quantum phases |

---

## Pipeline Configuration Summary

```python
# Feature Selection Phase
- n_fireflies: 30
- max_iterations: 100
- attraction: 0.5 (adaptive)
- randomness: 0.3 (adaptive)
- quantum_factor: 0.1
- batch_size: 32
- early_stopping: 30 iterations no improvement

# K-Fold Evaluation Phase
- n_splits: 5
- epochs: 30 per fold
- batch_size: 32
- early_stopping patience: 10 epochs
- dropout: 0.3 (regularization)

# Feature Selection Output
- n_features_to_select: 500
- original_dimension: ~2000+
- reduction_ratio: ~75%
```

---

## Performance Metrics Tracked

1. **Optimization Phase**
   - Best loss per iteration
   - Convergence curve

2. **Evaluation Phase**
   - K-Fold CV Accuracy: aggregate OOF predictions
   - Test Set Accuracy: holdout test performance
   - Classification Report: per-class metrics

3. **Model Checkpoints**
   - Best model per fold (saved based on val_accuracy)
   - Enables ensemble predictions if needed

