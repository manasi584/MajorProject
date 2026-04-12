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
# QUANTUM SUPERPOSITION MUTATION PUMA OPTIMIZER
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
# QUANTUM SUPERPOSITION MUTATION PUMA OPTIMIZER
# ==========================================

class QuantumPuma:
    """Quantum-enhanced puma with superposition mutation"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.fitness = float('inf')  # Energy/loss value
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.bounds = bounds

        # Quantum superposition state (qubits)
        self.qbit_state = np.random.uniform(0, 1, dim)  # Probability amplitudes
        self.phase = np.random.uniform(0, 2*np.pi, dim)  # Quantum phase

        # Puma behavioral state
        self.in_exploration = True  # Exploration vs exploitation phase
        self.energy_level = 1.0  # Puma energy (affects speed)

    def update_quantum_superposition(self, collapse_prob=0.3):
        """Update quantum superposition state (q-bit amplitudes and phase)"""
        # Quantum phase rotation
        phase_rotation = np.random.uniform(-np.pi/4, np.pi/4, self.phase.shape)
        self.phase = (self.phase + phase_rotation) % (2 * np.pi)

        # Q-bit amplitude update (Hadamard-like operation)
        hadamard_transform = (self.qbit_state + np.random.uniform(-0.1, 0.1, self.qbit_state.shape))
        self.qbit_state = np.abs(hadamard_transform)
        self.qbit_state = self.qbit_state / (np.sum(self.qbit_state) + 1e-8)  # Normalize

        # Quantum measurement (collapse with probability)
        if np.random.rand() < collapse_prob:
            # Collapse to basis state
            self.qbit_state = np.zeros_like(self.qbit_state)
            collapse_idx = np.random.choice(len(self.qbit_state),
                                          p=np.abs(self.qbit_state)**2 if np.sum(np.abs(self.qbit_state)**2) > 0 else None)
            self.qbit_state[collapse_idx] = 1.0

    def superposition_mutation(self, mutation_rate=0.1, iteration=0, max_iterations=100):
        """Apply quantum superposition mutation for diversity"""
        adaptive_rate = mutation_rate * (1 - iteration / max(iteration + max_iterations, 1))

        if np.random.rand() < adaptive_rate:
            # Quantum superposition-based mutation
            mutation_vector = (np.cos(self.phase) * self.qbit_state +
                             np.sin(self.phase) * (1 - self.qbit_state))

            # Apply mutation with adaptive intensity
            mutation_intensity = 0.05 * (1 - iteration / max(iteration + max_iterations, 1))
            self.position = self.position + mutation_intensity * mutation_vector

            # Enforce bounds
            self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

        self.update_quantum_superposition()

    def explore(self, bounds_range=1.0):
        """Exploration phase: broad search in solution space"""
        exploration_step = np.random.uniform(-bounds_range, bounds_range, self.position.shape)
        self.position = self.position + exploration_step * self.energy_level * 0.1
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def exploit(self, best_position, hunt_intensity=0.5):
        """Exploitation phase: hunt towards best solution (prey)"""
        direction = best_position - self.position
        hunt_step = hunt_intensity * direction * self.energy_level

        self.position = self.position + hunt_step
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def territorial_behavior(self, neighbor_positions, territory_radius=0.15):
        """Territorial behavior: maintain distance from neighbors"""
        if len(neighbor_positions) == 0:
            return

        for neighbor_pos in neighbor_positions:
            distance = np.linalg.norm(self.position - neighbor_pos) + 1e-8
            if distance < territory_radius:
                # Move away from neighbor
                direction = (self.position - neighbor_pos) / distance
                self.position = self.position + direction * 0.05

        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

    def update_energy(self, fitness_improvement, max_energy=1.0):
        """Update puma energy based on fitness improvement"""
        if fitness_improvement > 0:
            self.energy_level = min(max_energy, self.energy_level + 0.1)
        else:
            self.energy_level = max(0.1, self.energy_level - 0.1)

class QuantumSuperpositionMutationPumaOptimizer:
    """QSM-PO: Quantum Superposition Mutation Puma Optimizer with Hybrid Training"""
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_pumas=30, max_iterations=100, hunt_intensity=0.5,
                 exploration_rate=0.5, mutation_rate=0.15, batch_size=32, use_hybrid=True):
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
        self.use_hybrid = use_hybrid

        # Count total parameters
        self.param_count = sum(p.numel() for p in model.parameters())

        # Initialize puma population
        self.pumas = [QuantumPuma(self.param_count, bounds=(-0.1, 0.1))
                     for _ in range(n_pumas)]

        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.criterion = nn.MSELoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_fitness(self, position):
        """Evaluate puma fitness (model loss) with given weights"""
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

            # Validation loss as fitness
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
        """Run quantum superposition mutation puma optimization"""
        print("\n🐆 Quantum Superposition Mutation Puma Optimizer Starting...\n")
        print(f"   Population: {self.n_pumas}")
        print(f"   Iterations: {self.max_iterations}")
        print(f"   Hunt Intensity: {self.hunt_intensity}")
        print(f"   Exploration Rate: {self.exploration_rate}")
        print(f"   Mutation Rate: {self.mutation_rate}\n")

        # Evaluate initial puma population
        print("Evaluating initial puma population...")
        for puma in self.pumas:
            fitness = self.evaluate_fitness(puma.position)
            puma.fitness = fitness
            puma.best_fitness = fitness

            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = puma.position.copy()

        # Main optimization loop
        for iteration in range(self.max_iterations):
            # Adaptive phase change: exploration -> exploitation
            exploration_probability = self.exploration_rate * (1 - iteration / self.max_iterations)
            adaptive_hunt = self.hunt_intensity * (1 - 0.3 * iteration / self.max_iterations)
            adaptive_mutation = self.mutation_rate * (1 - iteration / self.max_iterations)

            # Update puma phases based on exploration probability
            for puma in self.pumas:
                puma.in_exploration = np.random.rand() < exploration_probability

            # Hunting hierarchy: alpha (best) leads the hunt
            sorted_indices = np.argsort([p.fitness for p in self.pumas])
            alpha_puma = self.pumas[sorted_indices[0]]
            beta_puma = self.pumas[sorted_indices[1]] if len(self.pumas) > 1 else alpha_puma

            # Update each puma
            for i, puma in enumerate(self.pumas):
                # Exploration or exploitation phase
                if puma.in_exploration:
                    # Exploration: broad search in solution space
                    puma.explore(bounds_range=1.0)
                else:
                    # Exploitation: hunt towards best puma
                    hunt_target = alpha_puma.position if np.random.rand() < 0.7 else beta_puma.position
                    puma.exploit(hunt_target, hunt_intensity=adaptive_hunt)

                # Territorial behavior: maintain distance from others
                nearby_idx = np.random.choice(len(self.pumas),
                                            size=max(1, len(self.pumas)//4),
                                            replace=False)
                nearby_positions = [self.pumas[idx].position for idx in nearby_idx if idx != i]
                puma.territorial_behavior(nearby_positions, territory_radius=0.15)

                # Apply quantum superposition mutation for diversity
                puma.superposition_mutation(
                    mutation_rate=adaptive_mutation,
                    iteration=iteration,
                    max_iterations=self.max_iterations
                )

                # Evaluate new position
                fitness = self.evaluate_fitness(puma.position)
                prev_fitness = puma.fitness
                puma.fitness = fitness

                # Update energy based on fitness improvement
                fitness_improvement = prev_fitness - fitness
                puma.update_energy(fitness_improvement)

                # Track personal best
                if fitness < puma.best_fitness:
                    puma.best_fitness = fitness
                    puma.best_position = puma.position.copy()

                # Track global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = puma.position.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.best_fitness_history.append(self.global_best_fitness)

            if (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                improvement = self.best_fitness_history[0] - self.global_best_fitness
                exploration_count = sum(1 for p in self.pumas if p.in_exploration)
                print(f"Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Fitness: {self.global_best_fitness:.6f} | "
                      f"Improvement: {improvement:.6f} | "
                      f"Exploring: {exploration_count}/{self.n_pumas}")

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
            val_fitness = self.evaluate_fitness(self._get_weights())

            if val_fitness < self.global_best_fitness:
                self.global_best_fitness = val_fitness
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

    # Run Quantum Superposition Mutation Puma Optimizer
    qsmpo = QuantumSuperpositionMutationPumaOptimizer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_pumas=30,
        max_iterations=150,
        hunt_intensity=0.5,
        exploration_rate=0.5,
        mutation_rate=0.15,
        batch_size=32,
        use_hybrid=True
    )

    qsmpo.optimize()

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
    print("🐆 QUANTUM PUMA OPTIMIZER RESULTS")
    print("=" * 40)
    print(f"MAE:     {mae:.3f}")
    print(f"RMSE:    {rmse:.3f}")
    print(f"R²:      {r2:.3f}")
    print(f"Pearson: {pearson_corr:.3f}")
    print("=" * 40)
