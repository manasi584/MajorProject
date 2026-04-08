# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "opencv-python>=4.13.0.92",
#     "pandas>=2.3.3",
#     "scikit-learn>=1.6.1",
#     "scipy>=1.13.1",
#     "torch>=2.8.0",
#     "torchvision>=0.23.0",
#     "tqdm>=4.67.3",
# ]
# ///

# ==========================================
# QUANTUM MUTATION REPTILE SEARCH ALGORITHM
# CLASSIFICATION (AQI) - ResNet50 only (2048-dim, no DCP)
# ==========================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cpu")

NUM_CLASSES = 6  # Good, Moderate, USG, Unhealthy, Very Unhealthy, Hazardous

# ==========================================
# AQI CLASS CONVERSION
# ==========================================

def pm25_to_classes(pm25_array):
    """Convert PM2.5 values to AQI class labels (0-5)"""
    thresholds = [12.0, 35.4, 55.4, 150.4, 250.4]
    return np.digitize(pm25_array, thresholds).astype(int)

# ==========================================
# FEATURE EXTRACTION (ResNet50 only, 2048-dim)
# ==========================================

def extract_features(dataset_path, csv_path, image_dir, pm25_max=None):
    """Extract ResNet50 features (2048-dim per image, no DCP)"""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    if pm25_max is None:
        pm25_max = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / pm25_max

    print("Initializing ResNet50 model...")
    backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()  # removes FC head, outputs 2048-dim
    backbone = backbone.to(device)
    backbone.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    features_list = []
    labels_list = []

    print("Extracting features...")
    for img_name, label in tqdm(zip(df['Filename'], df['PM2.5']), total=len(df)):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(img_tensor).cpu().numpy().flatten()

        features_list.append(feat)
        labels_list.append(label)

    X = np.array(features_list)   # shape: (N, 2048)
    y = np.array(labels_list)

    return X, y, pm25_max

# ==========================================
# QUANTUM MUTATION REPTILE SEARCH ALGORITHM
# ==========================================

class QuantumReptile:
    """Quantum-mutated reptile for crocodile-inspired optimization"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.energy = float('inf')
        self.best_position = self.position.copy()
        self.best_energy = float('inf')
        self.bounds = bounds
        self.quantum_phase = np.random.uniform(0, 2*np.pi, dim)
        self.qbit_alpha = np.random.uniform(0, 1, dim)
        self.qbit_beta = np.sqrt(1 - self.qbit_alpha**2)

    def update_quantum_mutation(self, step_size=0.1):
        self.quantum_phase += np.random.uniform(-step_size, step_size, self.quantum_phase.shape)
        self.quantum_phase = self.quantum_phase % (2 * np.pi)
        rotation_angle = np.random.uniform(0, 2*np.pi, self.qbit_alpha.shape)
        self.qbit_alpha = np.cos(rotation_angle)
        self.qbit_beta = np.sin(rotation_angle)

    def encircle_prey(self, prey_position, encircle_factor=0.5):
        coeff = 2 * encircle_factor * np.random.rand(*self.position.shape) - encircle_factor
        self.position = prey_position - coeff * (prey_position - self.position)

    def hunt_cooperatively(self, pack_positions, hunt_factor=0.3):
        pack_center = np.mean(pack_positions, axis=0) if len(pack_positions) > 0 else self.position
        hunt_coeff = hunt_factor * np.random.rand(*self.position.shape)
        self.position = self.position + hunt_coeff * (pack_center - self.position)

    def apply_quantum_mutation(self, mutation_rate=0.1, iteration=0, max_iterations=100):
        adaptive_mutation_rate = mutation_rate * (1 - iteration / max(iteration + max_iterations, 1))
        if np.random.rand() < adaptive_mutation_rate:
            quantum_mutation = (self.qbit_alpha * np.cos(self.quantum_phase) +
                                self.qbit_beta * np.sin(self.quantum_phase))
            mutation_intensity = (1 - iteration / max(iteration + max_iterations, 1)) * 0.05
            self.position = self.position + mutation_intensity * quantum_mutation
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        self.update_quantum_mutation()

class QuantumMutationReptileOptimizer:
    """Quantum Mutation Reptile Search Algorithm with Hybrid Training"""
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_reptiles=30, max_iterations=100, encircle_factor=0.5,
                 hunt_factor=0.3, mutation_rate=0.1, batch_size=32, use_hybrid=True):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_reptiles = n_reptiles
        self.max_iterations = max_iterations
        self.encircle_factor = encircle_factor
        self.hunt_factor = hunt_factor
        self.mutation_rate = mutation_rate
        self.batch_size = batch_size
        self.use_hybrid = use_hybrid

        self.param_count = sum(p.numel() for p in model.parameters())
        self.reptiles = [QuantumReptile(self.param_count, bounds=(-0.1, 0.1))
                        for _ in range(n_reptiles)]

        self.global_best_position = None
        self.global_best_energy = float('inf')
        self.criterion = nn.CrossEntropyLoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_energy(self, position):
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

    def optimize(self):
        print("\nQuantum Mutation Reptile Search Algorithm (ResNet50, no DCP) Starting...\n")
        print(f"   Population: {self.n_reptiles}")
        print(f"   Iterations: {self.max_iterations}")
        print(f"   Encircle Factor: {self.encircle_factor}")
        print(f"   Hunt Factor: {self.hunt_factor}")
        print(f"   Mutation Rate: {self.mutation_rate}\n")

        print("Evaluating initial reptile population...")
        for reptile in self.reptiles:
            energy = self.evaluate_energy(reptile.position)
            reptile.energy = energy
            reptile.best_energy = energy
            if energy < self.global_best_energy:
                self.global_best_energy = energy
                self.global_best_position = reptile.position.copy()

        for iteration in range(self.max_iterations):
            adaptive_encircle = self.encircle_factor * (1 - iteration / self.max_iterations)
            adaptive_hunt = self.hunt_factor * (1 - 0.5 * iteration / self.max_iterations)
            adaptive_mutation = self.mutation_rate * (1 - iteration / self.max_iterations)

            sorted_indices = np.argsort([r.energy for r in self.reptiles])
            alpha_reptile = self.reptiles[sorted_indices[0]]
            beta_reptile = self.reptiles[sorted_indices[1]] if len(self.reptiles) > 1 else alpha_reptile

            for i, reptile in enumerate(self.reptiles):
                if np.random.rand() < 0.5:
                    reptile.encircle_prey(alpha_reptile.position, adaptive_encircle)
                else:
                    reptile.encircle_prey(beta_reptile.position, adaptive_encircle)

                nearby_idx = np.random.choice(len(self.reptiles),
                                              size=max(1, len(self.reptiles)//3),
                                              replace=False)
                nearby_positions = [self.reptiles[idx].position for idx in nearby_idx]
                reptile.hunt_cooperatively(nearby_positions, adaptive_hunt)

                reptile.apply_quantum_mutation(
                    mutation_rate=adaptive_mutation,
                    iteration=iteration,
                    max_iterations=self.max_iterations
                )

                energy = self.evaluate_energy(reptile.position)
                reptile.energy = energy

                if energy < reptile.best_energy:
                    reptile.best_energy = energy
                    reptile.best_position = reptile.position.copy()

                if energy < self.global_best_energy:
                    self.global_best_energy = energy
                    self.global_best_position = reptile.position.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.best_fitness_history.append(self.global_best_energy)

            if (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                improvement = self.best_fitness_history[0] - self.global_best_energy
                print(f"Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Energy: {self.global_best_energy:.6f} | "
                      f"Total Improvement: {improvement:.6f}")

            if self.no_improvement_count > 20:
                print(f"\nEarly stopping at iteration {iteration+1} (no improvement)")
                break

        if self.use_hybrid:
            print("\nHybrid Refinement: Fine-tuning with gradient descent...")
            self._hybrid_gradient_refinement()

        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype
            )
            idx += param_size

        print("\nOptimization Complete!\n")

    def _hybrid_gradient_refinement(self, iterations=10):
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype
            )
            param.requires_grad = True
            idx += param_size

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for _ in range(iterations):
            self.model.train()
            for j in range(0, len(self.X_train), self.batch_size):
                X_batch = torch.tensor(self.X_train[j:j+self.batch_size], dtype=torch.float32)
                y_batch = torch.tensor(self.y_train[j:j+self.batch_size], dtype=torch.long)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            val_energy = self.evaluate_energy(self._get_weights())
            if val_energy < self.global_best_energy:
                self.global_best_energy = val_energy
                self.global_best_position = self._get_weights().copy()

    def _get_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

# ==========================================
# MODEL DEFINITION
# ==========================================

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    DATASET_PATH = "./data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset"
    CSV_PATH = os.path.join(DATASET_PATH, "IND_and_Nep_AQI_Dataset.csv")
    IMAGE_DIR = os.path.join(DATASET_PATH, "All_img")
    FEATURES_PATH = "./air_pollution_features_resnet50.npz"

    if os.path.exists(FEATURES_PATH):
        print("Loading pre-extracted ResNet50 features...")
        data = np.load(FEATURES_PATH)
        X = data['X']
        y = data['y']
        PM25_MAX = float(data['PM25_MAX'])
    else:
        print("Extracting ResNet50 features (2048-dim)...")
        X, y, PM25_MAX = extract_features(DATASET_PATH, CSV_PATH, IMAGE_DIR)
        np.savez(FEATURES_PATH, X=X, y=y, PM25_MAX=PM25_MAX)
        print(f"Features saved to {FEATURES_PATH}")

    y_raw = y * PM25_MAX
    y_class = pm25_to_classes(y_raw)
    print(f"Class distribution: {np.bincount(y_class)}")
    print(f"Input feature dim: {X.shape[1]}")  # 2048

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_class, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    model = SimpleModel(X_train.shape[1]).to(device)

    qmrso = QuantumMutationReptileOptimizer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_reptiles=30,
        max_iterations=150,
        encircle_factor=0.5,
        hunt_factor=0.3,
        mutation_rate=0.1,
        batch_size=32,
        use_hybrid=True
    )

    qmrso.optimize()

    print("\nEVALUATION ON TEST SET\n")
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 63)
    print("QUANTUM REPTILE (ResNet50, no DCP) - CLASSIFICATION RESULTS")
    print("=" * 63)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"],
        zero_division=0
    ))
    print("=" * 63)
