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
# TRAINING & FEATURE EXTRACTION
# ==========================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cpu")

# ==========================================
# FEATURE EXTRACTION (Dark Channel Prior + ResNet18)
# ==========================================

def dark_channel(image, size=15):
    """Extract dark channel prior from image"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return np.mean(dark)

def extract_features(dataset_path, csv_path, image_dir, pm25_max=None):
    """Extract ResNet18 + DCP features"""
    print("📊 Loading dataset...")
    df = pd.read_csv(csv_path)

    if pm25_max is None:
        pm25_max = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / pm25_max

    print("🔧 Initializing ResNet18 model...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    features_list = []
    dcp_list = []
    labels_list = []

    print("🚀 Extracting features...")
    for img_name, label in tqdm(zip(df['Filename'], df['PM2.5']), total=len(df)):
        img_path = os.path.join(image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dcp = dark_channel(image)

        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().flatten()

        features_list.append(feat)
        dcp_list.append(dcp)
        labels_list.append(label)

    X = np.array(features_list)
    DCP = np.array(dcp_list).reshape(-1, 1)
    y = np.array(labels_list)
    DCP = (DCP - DCP.mean()) / (DCP.std() + 1e-8)

    return X, DCP, y, pm25_max

# ==========================================
# QUANTUM MUTATION REPTILE SEARCH ALGORITHM
# ==========================================

class QuantumReptile:
    """Quantum-mutated reptile for crocodile-inspired optimization"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.energy = float('inf')  # Fitness (lower is better)
        self.best_position = self.position.copy()
        self.best_energy = float('inf')
        self.bounds = bounds
        # Quantum phase for mutation encoding
        self.quantum_phase = np.random.uniform(0, 2*np.pi, dim)
        # Q-bit representation for probabilistic behavior
        self.qbit_alpha = np.random.uniform(0, 1, dim)
        self.qbit_beta = np.sqrt(1 - self.qbit_alpha**2)

    def update_quantum_mutation(self, step_size=0.1):
        """Update quantum phase and q-bit states"""
        # Quantum phase rotation
        self.quantum_phase += np.random.uniform(-step_size, step_size, self.quantum_phase.shape)
        self.quantum_phase = self.quantum_phase % (2 * np.pi)

        # Q-bit amplitude rotation
        rotation_angle = np.random.uniform(0, 2*np.pi, self.qbit_alpha.shape)
        self.qbit_alpha = np.cos(rotation_angle)
        self.qbit_beta = np.sin(rotation_angle)

    def encircle_prey(self, prey_position, encircle_factor=0.5):
        """Encircle mechanism: move towards prey (best solution)"""
        distance = np.linalg.norm(self.position - prey_position) + 1e-8

        # Adaptive encircling coefficient (decreases over time)
        coeff = 2 * encircle_factor * np.random.rand(*self.position.shape) - encircle_factor

        # Update position by encircling
        self.position = prey_position - coeff * (prey_position - self.position)

    def hunt_cooperatively(self, pack_positions, hunt_factor=0.3):
        """Cooperative hunting: coordinate with other reptiles"""
        # Average position of nearby reptiles
        pack_center = np.mean(pack_positions, axis=0) if len(pack_positions) > 0 else self.position

        # Move towards pack center with hunting cooperation
        hunt_coeff = hunt_factor * np.random.rand(*self.position.shape)
        self.position = self.position + hunt_coeff * (pack_center - self.position)

    def apply_quantum_mutation(self, mutation_rate=0.1, iteration=0, max_iterations=100):
        """Apply quantum mutation for diversity enhancement"""
        # Mutation probability decreases over iterations
        adaptive_mutation_rate = mutation_rate * (1 - iteration / max(iteration + max_iterations, 1))

        if np.random.rand() < adaptive_mutation_rate:
            # Quantum mutation using q-bit and phase
            quantum_mutation = (self.qbit_alpha * np.cos(self.quantum_phase) +
                               self.qbit_beta * np.sin(self.quantum_phase))

            # Apply mutation with adaptive intensity
            mutation_intensity = (1 - iteration / max(iteration + max_iterations, 1)) * 0.05
            self.position = self.position + mutation_intensity * quantum_mutation

        # Enforce bounds
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

        # Count total parameters
        self.param_count = sum(p.numel() for p in model.parameters())

        # Initialize reptile population
        self.reptiles = [QuantumReptile(self.param_count, bounds=(-0.1, 0.1))
                        for _ in range(n_reptiles)]

        self.global_best_position = None
        self.global_best_energy = float('inf')
        self.criterion = nn.MSELoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_energy(self, position):
        """Evaluate reptile energy (fitness) with given weights"""
        try:
            # Set model weights from position vector
            idx = 0
            for param in self.model.parameters():
                param_size = param.numel()
                param.data = torch.tensor(
                    position[idx:idx+param_size].reshape(param.shape),
                    dtype=param.dtype
                )
                idx += param_size

            # Validation loss as energy (fitness)
            self.model.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(self.X_val), self.batch_size):
                    X_batch = torch.tensor(self.X_val[i:i+self.batch_size], dtype=torch.float32)
                    y_batch = torch.tensor(self.y_val[i:i+self.batch_size], dtype=torch.float32)

                    outputs = self.model(X_batch).squeeze()
                    loss = self.criterion(outputs, y_batch)
                    total_loss += loss.item()
                    num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            return avg_loss
        except Exception:
            return float('inf')

    def optimize(self):
        """Run quantum mutation reptile search optimization"""
        print("\n🦎 Quantum Mutation Reptile Search Algorithm Starting...\n")
        print(f"   Population: {self.n_reptiles}")
        print(f"   Iterations: {self.max_iterations}")
        print(f"   Encircle Factor: {self.encircle_factor}")
        print(f"   Hunt Factor: {self.hunt_factor}")
        print(f"   Mutation Rate: {self.mutation_rate}\n")

        # Evaluate initial reptiles
        print("Evaluating initial reptile population...")
        for idx, reptile in enumerate(self.reptiles):
            energy = self.evaluate_energy(reptile.position)
            reptile.energy = energy
            reptile.best_energy = energy

            if energy < self.global_best_energy:
                self.global_best_energy = energy
                self.global_best_position = reptile.position.copy()

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Adaptive parameters
            adaptive_encircle = self.encircle_factor * (1 - iteration / self.max_iterations)
            adaptive_hunt = self.hunt_factor * (1 - 0.5 * iteration / self.max_iterations)
            adaptive_mutation = self.mutation_rate * (1 - iteration / self.max_iterations)

            # Find best (alpha) and second-best (beta) reptiles
            sorted_indices = np.argsort([r.energy for r in self.reptiles])
            alpha_reptile = self.reptiles[sorted_indices[0]]
            beta_reptile = self.reptiles[sorted_indices[1]] if len(self.reptiles) > 1 else alpha_reptile

            # Update each reptile
            for i, reptile in enumerate(self.reptiles):
                # Encircle prey (best solution)
                if np.random.rand() < 0.5:
                    # Encircle alpha (best)
                    reptile.encircle_prey(alpha_reptile.position, adaptive_encircle)
                else:
                    # Encircle beta (second best)
                    reptile.encircle_prey(beta_reptile.position, adaptive_encircle)

                # Cooperative hunting with nearby reptiles
                nearby_idx = np.random.choice(len(self.reptiles),
                                            size=max(1, len(self.reptiles)//3),
                                            replace=False)
                nearby_positions = [self.reptiles[idx].position for idx in nearby_idx]
                reptile.hunt_cooperatively(nearby_positions, adaptive_hunt)

                # Apply quantum mutation for diversity
                reptile.apply_quantum_mutation(
                    mutation_rate=adaptive_mutation,
                    iteration=iteration,
                    max_iterations=self.max_iterations
                )

                # Evaluate new position
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

            # Early stopping if no improvement
            if self.no_improvement_count > 20:
                print(f"\n⚠️  Early stopping at iteration {iteration+1} (no improvement)")
                break

        # Hybrid refinement: Fine-tune with gradient descent
        if self.use_hybrid:
            print("\n🔧 Hybrid Refinement: Fine-tuning with gradient descent...")
            self._hybrid_gradient_refinement()

        # Set model to best weights
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype
            )
            idx += param_size

        print("\n✅ Optimization Complete!\n")

    def _hybrid_gradient_refinement(self, iterations=10):
        """Fine-tune best solution with gradient descent"""
        # Set model to best position
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
            loss_total = 0

            for j in range(0, len(self.X_train), self.batch_size):
                X_batch = torch.tensor(self.X_train[j:j+self.batch_size], dtype=torch.float32)
                y_batch = torch.tensor(self.y_train[j:j+self.batch_size], dtype=torch.float32)

                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                loss_total += loss.item()

            # Evaluate on validation set
            val_energy = self.evaluate_energy(self._get_weights())

            if val_energy < self.global_best_energy:
                self.global_best_energy = val_energy
                self.global_best_position = self._get_weights().copy()

    def _get_weights(self):
        """Extract weights as flat vector"""
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
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Paths
    DATASET_PATH = "./data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset"
    CSV_PATH = os.path.join(DATASET_PATH, "IND_and_Nep_AQI_Dataset.csv")
    IMAGE_DIR = os.path.join(DATASET_PATH, "All_img")
    FEATURES_PATH = "./air_pollution_features.npz"

    # Load or extract features
    if os.path.exists(FEATURES_PATH):
        print("📂 Loading pre-extracted features...")
        data = np.load(FEATURES_PATH)
        X = data['X']
        DCP = data['DCP']
        y = data['y']
        PM25_MAX = data['PM25_MAX']
    else:
        print("🔨 Extracting features...")
        X, DCP, y, PM25_MAX = extract_features(DATASET_PATH, CSV_PATH, IMAGE_DIR)
        np.savez(FEATURES_PATH, X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)

    # Combine features
    X_combined = np.concatenate([X, DCP], axis=1)

    # Split data: 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Initialize model
    model = SimpleModel(X_train.shape[1]).to(device)

    # Run Quantum Mutation Reptile Search Optimization
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

    # Evaluate on test set
    print("\n📊 EVALUATION ON TEST SET\n")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_test_tensor).squeeze().cpu().numpy()
        y_pred = outputs
        y_true = y_test

    # Denormalize
    y_true = np.array(y_true) * PM25_MAX
    y_pred = np.array(y_pred) * PM25_MAX

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print("=" * 40)
    print("🦎 QUANTUM REPTILE SEARCH RESULTS")
    print("=" * 40)
    print(f"MAE:     {mae:.3f}")
    print(f"RMSE:    {rmse:.3f}")
    print(f"R²:      {r2:.3f}")
    print(f"Pearson: {pearson_corr:.3f}")
    print("=" * 40)
