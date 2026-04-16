# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "matplotlib>=3.9.4",
#     "numpy>=2.0.2",
#     "scikit-learn>=1.6.1",
#     "tensorflow>=2.15.0",
#     "torch>=2.8.0",
#     "torchvision>=0.23.0",
# ]
# ///

# ==========================================
# K-FOLD EVALUATION WITH FEATURE SELECTION
# ==========================================

import os
import sys
import random
import importlib.util
import logging

# Suppress TensorFlow/Keras warnings about HDF5 format
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ==========================================
# UTILITIES
# ==========================================

def set_seed(seed=42):
    """Set seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

class Logger:
    """Logs to both console and file"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def close(self):
        self.log.close()

def pm25_to_classes(pm25_array):
    """Convert PM2.5 values to AQI class labels (0-5)"""
    thresholds = [12.0, 35.4, 55.4, 150.4, 250.4]
    return np.digitize(pm25_array, thresholds).astype(int)

def load_features(features_path):
    """Load extracted features from .npz file"""
    print(f"Loading features from {features_path}...")
    data = np.load(features_path)
    X = data['X']
    DCP = data['DCP']
    y = data['y']
    PM25_MAX = float(data.get('PM25_MAX', 1.0))

    print(f"  Features shape: {X.shape}")
    print(f"  DCP shape: {DCP.shape}")
    print(f"  Labels shape: {y.shape}")

    return X, DCP, y, PM25_MAX


# ==========================================
# FEATURE SELECTION WITH QUANTUM OPTIMIZER
# ==========================================

def select_features(X_train, y_train, X_val, y_val, optimizer_class, optimizer_params,
                   n_features, pm25_max=1.0, device=None):
    """
    Use a quantum optimizer to rank features by importance.
    Creates a small PyTorch model, runs the quantum optimizer,
    then ranks features by input layer weight magnitudes.

    Args:
        X_train: Training features
        y_train: Training labels (normalized PM2.5 values)
        X_val: Validation features
        y_val: Validation labels (normalized PM2.5 values)
        optimizer_class: Quantum optimizer class (e.g., QuantumSuperpositionMutationPumaOptimizer)
        optimizer_params: Dictionary of optimizer parameters
        n_features: Number of features to select
        pm25_max: PM2.5 max value for denormalization
        device: PyTorch device

    Returns:
        selected_indices: Array of selected feature indices (sorted)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print(f"FEATURE SELECTION - {optimizer_class.__name__}")
    print("="*60)

    # Convert PM2.5 to class labels
    y_train_denorm = y_train * pm25_max
    y_val_denorm = y_val * pm25_max
    y_train_int = pm25_to_classes(y_train_denorm).astype(int)
    y_val_int = pm25_to_classes(y_val_denorm).astype(int)

    # Simple PyTorch model for feature ranking
    class FeatureSelectorModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 6)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = FeatureSelectorModel(X_train.shape[1]).to(device)

    # Setup optimizer parameters
    params = optimizer_params.copy()
    params.update({
        'model': model,
        'X_train': X_train,
        'y_train': y_train_int,
        'X_val': X_val,
        'y_val': y_val_int,
        'device': device
    })

    # Run quantum optimizer
    optimizer = optimizer_class(**params)
    optimizer.optimize(verbose=True)

    # Extract feature importance from input layer weights
    with torch.no_grad():
        # Sum of absolute weights across output units for each input feature
        importance = model.fc1.weight.abs().sum(dim=0).cpu().numpy()

    # Select top N features
    selected = np.argsort(importance)[::-1][:n_features]
    selected = np.sort(selected)  # Sort indices for reproducibility

    print(f"\n✓ Selected {len(selected)} features (from {X_train.shape[1]})")
    print(f"  Top 5 feature indices: {selected[:5]}")
    print("="*60 + "\n")

    return selected


# ==========================================
# MODEL DEFINITION
# ==========================================

def build_simple_model(input_dim, num_classes=6):
    """Build simple fully-connected model (matches step 02)"""
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==========================================
# K-FOLD EVALUATION
# ==========================================

def compute_roc_auc(y_test, test_preds, results_dir="./results", feature_name="", feature_selection=""):
    """
    Compute ROC-AUC for multiclass classification and plot curves

    Args:
        y_test: Test labels (class indices)
        test_preds: Prediction probabilities (n_samples, n_classes)
        results_dir: Directory to save plots
        feature_name: Feature extraction method name
        feature_selection: Feature selection method name
    """
    n_classes = test_preds.shape[1]

    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    class_names = ["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]

    print("\nROC-AUC Scores per class:")
    print("-" * 40)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f"  {class_names[i]:20s}: {roc_auc[i]:.4f}")

    # Compute macro-average ROC-AUC
    macro_auc = np.mean(list(roc_auc.values()))
    print("-" * 40)
    print(f"  {'Macro-average':20s}: {macro_auc:.4f}")

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    roc_filename = f"{feature_name}_{feature_selection}.png"
    roc_plot_path = os.path.join(results_dir, roc_filename)
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ ROC curve plot saved: {roc_plot_path}")

    return roc_auc, macro_auc


def kfold_evaluation(X_features, y_labels, n_splits=5, epochs=30, batch_size=32,
                     seed=42, checkpoint_dir="./models", pm25_max=1.0, selected_indices=None,
                     results_dir="./results", feature_name="", feature_selection=""):
    """
    Perform k-fold cross-validation with simple FC model

    Args:
        X_features: Feature matrix (n_samples, n_features)
        y_labels: Labels (n_samples,) - normalized PM2.5 values (0-1)
        n_splits: Number of k-fold splits
        epochs: Number of training epochs
        batch_size: Batch size for training
        seed: Random seed
        checkpoint_dir: Directory to save model checkpoints
        pm25_max: Max PM2.5 value for denormalization
        selected_indices: Array of selected feature indices (None = use all features)
        results_dir: Directory to save results and plots
        feature_name: Feature extraction method name
        feature_selection: Feature selection method name

    Returns:
        cv_score: Cross-validation accuracy
        test_score: Test accuracy
        y_test: Test labels
        y_pred: Test predictions
        roc_auc: Dictionary of ROC-AUC scores per class
        macro_auc: Macro-average ROC-AUC
    """

    # Use selected features if provided
    if selected_indices is not None:
        X_features = X_features[:, selected_indices]
        print(f"\nUsing {len(selected_indices)} selected features\n")

    print("\n" + "="*60)
    print(f"K-FOLD EVALUATION ({n_splits}-fold CV)")
    print("="*60)

    # Denormalize PM2.5 values before converting to classes
    y_denormalized = y_labels * pm25_max
    y_class = pm25_to_classes(y_denormalized)
    print(f"\nClass distribution: {np.bincount(y_class)}")

    # Train-test split (80-20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_features, y_class, test_size=0.2, random_state=seed, stratify=y_class
    )

    print(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")

    # K-fold setup
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds = np.zeros((len(X_train_full), 6))
    test_preds = np.zeros((len(X_test), 6))

    print(f"\nStarting {n_splits}-fold CV...\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"\n🔥 Fold {fold+1}/{n_splits}")

        X_tr, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
        y_tr, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

        # Create model
        model = build_simple_model(X_train_full.shape[1])

        # Callbacks
        checkpoint = ModelCheckpoint(
            filepath=f"{checkpoint_dir}/best_model_fold_{fold}.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )

        # Train model
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stop],
            verbose=0
        )

        # OOF predictions
        val_pred = model.predict(X_val_fold, verbose=0)
        oof_preds[val_idx] = val_pred

        # Test predictions
        test_pred = model.predict(X_test, verbose=0)
        test_preds += test_pred / n_splits

        print(f"  ✓ Fold {fold+1} complete (epochs: {len(history.history['loss'])})")

    # Compute metrics
    oof_labels = np.argmax(oof_preds, axis=1)
    cv_score = accuracy_score(y_train_full, oof_labels)

    test_labels = np.argmax(test_preds, axis=1)
    test_score = accuracy_score(y_test, test_labels)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"K-FOLD CV Accuracy:  {cv_score:.4f} ({cv_score*100:.2f}%)")
    print(f"TEST ACCURACY:       {test_score:.4f} ({test_score*100:.2f}%)")
    print("="*60)

    print("\nClassification Report (Test Set):")
    print(classification_report(
        y_test, test_labels,
        target_names=["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"],
        zero_division=0,
        digits=4
    ))

    # Compute ROC-AUC
    roc_auc, macro_auc = compute_roc_auc(y_test, test_preds, results_dir, feature_name, feature_selection)

    return cv_score, test_score, y_test, test_labels, roc_auc, macro_auc

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # Configuration
    FEATURES_PATH = "./features_efficientnet.npz"  # or features_resnet50.npz
    FEATURE_SELECTION = "quantum_firefly"  # Options: "none", "quantum_puma", "quantum_firefly", "quantum_reptile", "firefly"
    N_FEATURES = 500  # Number of features to select (None = keep all)

    N_SPLITS = 5
    EPOCHS = 30
    BATCH_SIZE = 32
    SEED = 42
    CHECKPOINT_DIR = "./models"
    RESULTS_DIR = "./results"

    # Create results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(SEED)

    # Construct log filename from FEATURES_PATH and FEATURE_SELECTION
    feature_name = os.path.basename(FEATURES_PATH).replace(".npz", "")
    log_filename = f"{feature_name}_{FEATURE_SELECTION}.txt"
    log_filepath = os.path.join(RESULTS_DIR, log_filename)

    # Redirect stdout to log file (overwrites if exists)
    logger = Logger(log_filepath)
    original_stdout = sys.stdout
    sys.stdout = logger

    print(f"\n{'='*60}")
    print(f"Configuration: K-FOLD CV WITH SIMPLE FC MODEL")
    print(f"{'='*60}")
    print(f"Features:        {FEATURES_PATH}")
    print(f"Feature Selection: {FEATURE_SELECTION}")
    print(f"K-Folds:         {N_SPLITS}")
    print(f"Epochs:          {EPOCHS}")
    print(f"Batch Size:      {BATCH_SIZE}")
    print(f"Results file:    {log_filepath}")

    # Load features
    X, DCP, y, PM25_MAX = load_features(FEATURES_PATH)

    # Combine DCP with main features
    X_combined = np.concatenate([X, DCP], axis=1)
    original_dim = X_combined.shape[1]
    print(f"\nCombined features shape: {X_combined.shape}")

    # Feature selection with quantum optimizer
    selected_indices = None
    if FEATURE_SELECTION != "none" and N_FEATURES is not None:
        print(f"\nInitializing feature selection with {FEATURE_SELECTION}...")

        # Train-val split for feature selection (use 20% for validation)
        X_sel_train, X_sel_val, y_sel_train, y_sel_val = train_test_split(
            X_combined, y, test_size=0.2, random_state=SEED, stratify=pm25_to_classes(y * PM25_MAX)
        )

        # Import optimizer class
        if FEATURE_SELECTION == "quantum_puma":
            spec = importlib.util.spec_from_file_location(
                "qpuma", "./02_feature_selection_quantum_puma.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            optimizer_class = mod.QuantumSuperpositionMutationPumaOptimizer
            optimizer_params = {
                'n_pumas': 30,
                'max_iterations': 100,
                'hunt_intensity': 0.5,
                'exploration_rate': 0.5,
                'mutation_rate': 0.15,
                'batch_size': 32
            }
        elif FEATURE_SELECTION == "quantum_firefly":
            spec = importlib.util.spec_from_file_location(
                "qfirefly", "./02_feature_selection_quantum_firefly.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            optimizer_class = mod.QuantumFireflyOptimizer
            optimizer_params = {
                'n_fireflies': 30,
                'max_iterations': 100,
                'attraction': 0.5,
                'randomness': 0.3,
                'quantum_factor': 0.1,
                'batch_size': 32
            }
        elif FEATURE_SELECTION == "quantum_reptile":
            spec = importlib.util.spec_from_file_location(
                "qreptile", "./02_feature_selection_quantum_reptile.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            optimizer_class = mod.QuantumMutationReptileOptimizer
            optimizer_params = {
                'n_reptiles': 30,
                'max_iterations': 100,
                'encircle_factor': 0.5,
                'hunt_factor': 0.3,
                'mutation_rate': 0.1,
                'batch_size': 32
            }
        elif FEATURE_SELECTION == "firefly":
            spec = importlib.util.spec_from_file_location(
                "firefly", "./02_feature_selection_firefly.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            optimizer_class = mod.FireflyOptimizer
            optimizer_params = {
                'n_fireflies': 30,
                'max_iterations': 100,
                'attraction': 0.5,
                'randomness': 0.3,
                'batch_size': 32
            }
        else:
            raise ValueError(f"Unknown feature selection method: {FEATURE_SELECTION}")

        # Run feature selection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected_indices = select_features(
            X_sel_train, y_sel_train,
            X_sel_val, y_sel_val,
            optimizer_class,
            optimizer_params,
            N_FEATURES,
            pm25_max=PM25_MAX,
            device=device
        )

    # Run evaluation
    cv_score, test_score, y_test, y_pred, roc_auc, macro_auc = kfold_evaluation(
        X_combined, y,
        n_splits=N_SPLITS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        checkpoint_dir=CHECKPOINT_DIR,
        pm25_max=PM25_MAX,
        selected_indices=selected_indices,
        results_dir=RESULTS_DIR,
        feature_name=feature_name,
        feature_selection=FEATURE_SELECTION
    )

    # Close logger and restore stdout
    logger.close()
    sys.stdout = original_stdout
    print(f"✓ Results saved to: {log_filepath}")
