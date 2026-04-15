# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.4.4",
#     "opencv-python>=4.13.0.92",
#     "pandas>=3.0.2",
#     "scikit-learn>=1.8.0",
#     "torch>=2.11.0",
#     "torchvision>=0.26.0",
#     "tqdm>=4.67.3",
# ]
# ///

# ==========================================
# QUANTUM PUMA + MOBILENET CLASSIFICATION
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
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 6

# ==========================================
# AQI CLASS CONVERSION
# ==========================================

def pm25_to_classes(pm25_array):
    """Convert PM2.5 values to AQI class labels (0-5)"""
    thresholds = [12.0, 35.4, 55.4, 150.4, 250.4]
    return np.digitize(pm25_array, thresholds).astype(int)

# ==========================================
# FEATURE EXTRACTION (Dark Channel Prior + MobileNet)
# ==========================================

def dark_channel(image, size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return np.mean(dark)

def extract_features(dataset_path, csv_path, image_dir, pm25_max=None):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    if pm25_max is None:
        pm25_max = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / pm25_max

    print("Initializing MobileNetV2 model...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier = nn.Identity()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    features_list = []
    dcp_list = []
    labels_list = []

    print("Extracting features...")
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
# QUANTUM PUMA CLASS
# ==========================================

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

# ==========================================
# QUANTUM PUMA OPTIMIZER
# ==========================================

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
                    dtype=param.dtype,
                    device=device
                )
                idx += param_size

            self.model.eval()
            total_loss = 0
            num_batches = 0

            with torch.no_grad():
                for i in range(0, len(self.X_val), self.batch_size):
                    X_batch = torch.tensor(self.X_val[i:i+self.batch_size], dtype=torch.float32).to(device)
                    y_batch = torch.tensor(self.y_val[i:i+self.batch_size], dtype=torch.long).to(device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    total_loss += loss.item()
                    num_batches += 1

            return total_loss / max(num_batches, 1)
        except Exception:
            return float('inf')

    def optimize(self):
        print("\nQuantum Puma Optimizer (MobileNet) Starting...\n")
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

            if (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                improvement = self.best_fitness_history[0] - self.global_best_fitness
                exploration_count = sum(1 for p in self.pumas if p.in_exploration)
                print(f"Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Fitness: {self.global_best_fitness:.6f} | "
                      f"Improvement: {improvement:.6f}")

            if self.no_improvement_count > 20:
                print(f"\nEarly stopping at iteration {iteration+1} (no improvement)")
                break

        # Apply best weights
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype,
                device=device
            )
            idx += param_size

        print("\nOptimization Complete!\n")

    def _get_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

# ==========================================
# MODEL DEFINITION
# ==========================================

class MobileNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
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
    FEATURES_PATH = "./mobilenet_air_pollution_features.npz"

    if os.path.exists(FEATURES_PATH):
        print("Loading pre-extracted features...")
        data = np.load(FEATURES_PATH)
        X = data['X']
        DCP = data['DCP']
        y = data['y']
        PM25_MAX = float(data['PM25_MAX'])
    else:
        print("Extracting features...")
        X, DCP, y, PM25_MAX = extract_features(DATASET_PATH, CSV_PATH, IMAGE_DIR)
        np.savez(FEATURES_PATH, X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)

    y_raw = y * PM25_MAX
    y_class = pm25_to_classes(y_raw)
    print(f"Class distribution: {np.bincount(y_class)}")

    X_combined = np.concatenate([X, DCP], axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y_class, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    model = MobileNetClassifier(X_train.shape[1], NUM_CLASSES).to(device)

    optimizer = QuantumSuperpositionMutationPumaOptimizer(
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
        batch_size=32
    )

    optimizer.optimize()

    print("\nEVALUATION ON TEST SET\n")
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 50)
    print("QUANTUM PUMA + MOBILENET - RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"],
        zero_division=0
    ))
    print("=" * 50)
