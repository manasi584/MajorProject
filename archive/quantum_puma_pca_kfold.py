# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.4.4",
#     "pandas>=3.0.2",
#     "scikit-learn>=1.8.0",
#     "scipy>=1.17.1",
#     "torch>=2.11.0",
#     "tqdm>=4.67.3",
# ]
# ///

# ==========================================
# QUANTUM PUMA + PCA FEATURES + K-FOLD CV
# ==========================================

import math
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

BASE_DIR = "/Users/manasi/Desktop/MajorProject"
FEATURES_PATH = os.path.join(BASE_DIR, "air_pollution_features.npz")

# =========================
# CONFIG
# =========================
N_SPLITS = 5
N_PUMAS = 30
MAX_ITERATIONS = 100
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cpu")
NUM_CLASSES = 6

# =========================
# LOAD FEATURES
# =========================
print("Loading air pollution features...")
data = np.load(FEATURES_PATH)
X = data['X']
DCP = data['DCP']
y = data['y']

print(f"Features shape: {X.shape}")
print(f"DCP shape: {DCP.shape}")
print(f"Labels shape: {y.shape}")

# =========================
# PREPARE DATA
# =========================
# Combine features
X_combined = np.concatenate([X, DCP], axis=1)
print(f"Combined features shape: {X_combined.shape}")

# Flatten labels if needed (handle both 1D and 2D arrays)
if len(y.shape) > 1:
    y = y.flatten()

# Convert continuous PM2.5 values to class labels (0-5)
def pm25_to_classes(pm25_array):
    """Convert PM2.5 values to AQI class labels (0-5)"""
    thresholds = [12.0, 35.4, 55.4, 150.4, 250.4]
    return np.digitize(pm25_array, thresholds).astype(int)

y_class = pm25_to_classes(y)
print(f"Class distribution: {np.bincount(y_class)}")

# Split into train/val/test (60/20/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_combined, y_class, test_size=0.2, random_state=SEED, stratify=y_class
)
X_train_sel, X_val_sel, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=SEED, stratify=y_train_full
)

print(f"Train shape: {X_train_sel.shape}, Val shape: {X_val_sel.shape}, Test shape: {X_test.shape}")
print(f"Train labels: {y_train.shape}, Val labels: {y_val.shape}, Test labels: {y_test.shape}")

# X_full and y_full already prepared above

# =========================
# QUANTUM PUMA CLASS
# =========================
class QuantumPuma:
    """Quantum-enhanced puma with superposition mutation"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.fitness = float('inf')
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.bounds = bounds
        self.qbit_state = np.random.uniform(0, 1, dim)
        self.phase = np.random.uniform(0, 2*np.pi, dim)
        self.in_exploration = True
        self.energy_level = 1.0

    def update_quantum_superposition(self, collapse_prob=0.3):
        phase_rotation = np.random.uniform(-np.pi/4, np.pi/4, self.phase.shape)
        self.phase = (self.phase + phase_rotation) % (2 * np.pi)
        hadamard_transform = (self.qbit_state + np.random.uniform(-0.1, 0.1, self.qbit_state.shape))
        self.qbit_state = np.abs(hadamard_transform)
        self.qbit_state = self.qbit_state / (np.sum(self.qbit_state) + 1e-8)
        if np.random.rand() < collapse_prob:
            self.qbit_state = np.zeros_like(self.qbit_state)
            collapse_idx = np.random.choice(len(self.qbit_state),
                                            p=np.abs(self.qbit_state)**2 if np.sum(np.abs(self.qbit_state)**2) > 0 else None)
            self.qbit_state[collapse_idx] = 1.0

    def superposition_mutation(self, mutation_rate=0.1, iteration=0, max_iterations=100):
        adaptive_rate = mutation_rate * (1 - iteration / max(iteration + max_iterations, 1))
        if np.random.rand() < adaptive_rate:
            mutation_vector = (np.cos(self.phase) * self.qbit_state +
                               np.sin(self.phase) * (1 - self.qbit_state))
            mutation_intensity = 0.05 * (1 - iteration / max(iteration + max_iterations, 1))
            self.position = self.position + mutation_intensity * mutation_vector
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        self.update_quantum_superposition()

    def explore(self, bounds_range=1.0):
        exploration_step = np.random.uniform(-bounds_range, bounds_range, self.position.shape)
        self.position = self.position + exploration_step * self.energy_level * 0.1
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def exploit(self, best_position, hunt_intensity=0.5):
        direction = best_position - self.position
        hunt_step = hunt_intensity * direction * self.energy_level
        self.position = self.position + hunt_step
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def territorial_behavior(self, neighbor_positions, territory_radius=0.15):
        if len(neighbor_positions) == 0:
            return
        for neighbor_pos in neighbor_positions:
            distance = np.linalg.norm(self.position - neighbor_pos) + 1e-8
            if distance < territory_radius:
                direction = (self.position - neighbor_pos) / distance
                self.position = self.position + direction * 0.05
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def update_energy(self, fitness_improvement, max_energy=1.0):
        if fitness_improvement > 0:
            self.energy_level = min(max_energy, self.energy_level + 0.1)
        else:
            self.energy_level = max(0.1, self.energy_level - 0.1)

# =========================
# QUANTUM PUMA OPTIMIZER
# =========================
class QuantumSuperpositionMutationPumaOptimizer:
    """QSM-PO: Quantum Superposition Mutation Puma Optimizer"""
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_pumas=30, max_iterations=100, hunt_intensity=0.5,
                 exploration_rate=0.5, mutation_rate=0.15, batch_size=32):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_pumas = n_pumas
        self.max_iterations = max_iterations
        self.hunt_intensity = hunt_intensity
        self.exploration_rate = exploration_rate
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size

        self.param_count = sum(p.numel() for p in model.parameters())
        self.pumas = [QuantumPuma(self.param_count, bounds=(-0.1, 0.1))
                     for _ in range(n_pumas)]

        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.criterion = nn.CrossEntropyLoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_fitness(self, position):
        try:
            idx = 0
            for param in self.model.parameters():
                param_size = param.numel()
                param.data = torch.tensor(
                    position[idx:idx+param_size].reshape(param.shape),
                    dtype=param.dtype
                )
                idx += param_size

            self.model.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(self.X_val), self.batch_size):
                    X_batch = torch.tensor(self.X_val[i:i+self.batch_size], dtype=torch.float32)
                    y_batch = torch.tensor(self.y_val[i:i+self.batch_size], dtype=torch.long)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    total_loss += loss.item()
                    num_batches += 1

            return total_loss / max(num_batches, 1)
        except Exception:
            return float('inf')

    def optimize(self, verbose=True):
        if verbose:
            print("\n🔥 Quantum Puma Optimizer (K-Fold)")
            print(f"   Population: {self.n_pumas}")
            print(f"   Iterations: {self.max_iterations}")
            print(f"   Hunt Intensity: {self.hunt_intensity}\n")

        print("Evaluating initial puma population...")
        for puma in self.pumas:
            fitness = self.evaluate_fitness(puma.position)
            puma.fitness = fitness
            puma.best_fitness = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = puma.position.copy()

        for iteration in range(self.max_iterations):
            exploration_probability = self.exploration_rate * (1 - iteration / self.max_iterations)
            adaptive_hunt = self.hunt_intensity * (1 - 0.3 * iteration / self.max_iterations)
            adaptive_mutation = self.mutation_rate * (1 - iteration / self.max_iterations)

            for puma in self.pumas:
                puma.in_exploration = np.random.rand() < exploration_probability

            sorted_indices = np.argsort([p.fitness for p in self.pumas])
            alpha_puma = self.pumas[sorted_indices[0]]
            beta_puma = self.pumas[sorted_indices[1]] if len(self.pumas) > 1 else alpha_puma

            for i, puma in enumerate(self.pumas):
                if puma.in_exploration:
                    puma.explore(bounds_range=1.0)
                else:
                    hunt_target = alpha_puma.position if np.random.rand() < 0.7 else beta_puma.position
                    puma.exploit(hunt_target, hunt_intensity=adaptive_hunt)

                nearby_idx = np.random.choice(len(self.pumas),
                                              size=max(1, len(self.pumas)//4),
                                              replace=False)
                nearby_positions = [self.pumas[idx].position for idx in nearby_idx if idx != i]
                puma.territorial_behavior(nearby_positions, territory_radius=0.15)

                puma.superposition_mutation(
                    mutation_rate=adaptive_mutation,
                    iteration=iteration,
                    max_iterations=self.max_iterations
                )

                fitness = self.evaluate_fitness(puma.position)
                prev_fitness = puma.fitness
                puma.fitness = fitness
                puma.update_energy(prev_fitness - fitness)

                if fitness < puma.best_fitness:
                    puma.best_fitness = fitness
                    puma.best_position = puma.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = puma.position.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.best_fitness_history.append(self.global_best_fitness)

            if verbose and (iteration + 1) % max(1, self.max_iterations // 5) == 0:
                improvement = self.best_fitness_history[0] - self.global_best_fitness
                print(f"  Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Loss: {self.global_best_fitness:.6f}")

            if self.no_improvement_count > 15:
                if verbose:
                    print(f"  Early stopping at iteration {iteration+1}")
                break

        # Apply best weights
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype
            )
            idx += param_size

        if verbose:
            print("  Optimization Complete!\n")

    def _get_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

# =========================
# SIMPLE MODEL
# =========================
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# K-FOLD TRAINING
# =========================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

oof_preds = np.zeros((len(X_train_full), NUM_CLASSES))
test_preds = np.zeros((len(X_test), NUM_CLASSES))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):

    print(f"\n🚀 Fold {fold+1}/{N_SPLITS}")

    X_tr, X_val_fold = X_train_full[train_idx], X_train_full[val_idx]
    y_tr, y_val_fold = y_train_full[train_idx], y_train_full[val_idx]

    model = SimpleModel(X_train_full.shape[1]).to(DEVICE)

    optimizer = QuantumSuperpositionMutationPumaOptimizer(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val_fold,
        y_val=y_val_fold,
        n_pumas=N_PUMAS,
        max_iterations=MAX_ITERATIONS,
        hunt_intensity=0.5,
        exploration_rate=0.5,
        mutation_rate=0.15,
        batch_size=BATCH_SIZE
    )

    optimizer.optimize(verbose=False)

    # OOF predictions
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        val_pred = model(X_val_tensor).cpu().numpy()
    oof_preds[val_idx] = val_pred

    # Test predictions
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_pred = model(X_test_tensor).cpu().numpy()
    test_preds += test_pred / N_SPLITS

    print(f"  Fold {fold+1} Complete")

# =========================
# FINAL METRICS
# =========================
oof_labels = np.argmax(oof_preds, axis=1)
cv_score = accuracy_score(y_train_full, oof_labels)

print("\n" + "="*50)
print(f"🔥 K-FOLD CV Accuracy: {cv_score:.4f}")

test_labels = np.argmax(test_preds, axis=1)
test_score = accuracy_score(y_test, test_labels)

print(f"🔥 FINAL TEST ACCURACY: {test_score:.4f}")
print("="*50)

print("\nDetailed Classification Report:")
print(classification_report(y_test, test_labels, zero_division=0))
