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
# QUANTUM FIREFLY ALGORITHM
# CLASSIFICATION (AQI Class Prediction)
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
from torchvision.models import resnet18, ResNet18_Weights

device = torch.device("cpu")

NUM_CLASSES = 6  # Good, Moderate, USG, Unhealthy, Very Unhealthy, Hazardous

# ==========================================
# AQI CLASS CONVERSION
# ==========================================

def pm25_to_classes(pm25_array):
    """Convert PM2.5 values to AQI class labels (0-5)"""
    # Thresholds: Good=0-12, Moderate=12-35.4, USG=35.4-55.4,
    #             Unhealthy=55.4-150.4, Very Unhealthy=150.4-250.4, Hazardous=250.4+
    thresholds = [12.0, 35.4, 55.4, 150.4, 250.4]
    return np.digitize(pm25_array, thresholds).astype(int)

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
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    if pm25_max is None:
        pm25_max = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / pm25_max

    print("Initializing ResNet18 model...")
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
# QUANTUM FIREFLY ALGORITHM
# ==========================================

class QuantumFirefly:
    """Quantum-behaved firefly for swarm optimization"""
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.brightness = float('inf')
        self.best_position = self.position.copy()
        self.best_brightness = float('inf')
        self.bounds = bounds
        self.quantum_angle = np.random.uniform(0, 2*np.pi, dim)

    def update_quantum_angle(self, step_size=0.1):
        self.quantum_angle += np.random.uniform(-step_size, step_size, self.quantum_angle.shape)
        self.quantum_angle = self.quantum_angle % (2 * np.pi)

    def attract_to(self, other_firefly, attraction=0.5, randomness=0.3,
                   quantum_factor=0.1, iteration=0, max_iterations=100):
        distance = np.linalg.norm(self.position - other_firefly.position) + 1e-8
        beta0 = 1.0
        gamma = 1.0 / max(1.0, distance)
        beta = beta0 * np.exp(-gamma * distance ** 2)
        attraction_force = beta * (other_firefly.position - self.position)
        quantum_prob = quantum_factor * (1 - iteration / max(iteration + max_iterations, 1))
        if np.random.rand() < quantum_prob:
            self.position = np.random.uniform(self.bounds[0], self.bounds[1],
                                              self.position.shape)
        else:
            randomization = np.random.uniform(-1, 1, self.position.shape) * randomness
            quantum_modulation = np.sin(self.quantum_angle) * quantum_factor
            self.position = (self.position + attraction_force * attraction +
                             randomization + quantum_modulation)
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        self.update_quantum_angle()

class QuantumFireflyOptimizer:
    """Quantum-Inspired Firefly Algorithm with Hybrid Training"""
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_fireflies=30, max_iterations=100, attraction=0.5,
                 randomness=0.3, quantum_factor=0.1, batch_size=32, use_hybrid=True):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.attraction = attraction
        self.randomness = randomness
        self.quantum_factor = quantum_factor
        self.batch_size = batch_size
        self.use_hybrid = use_hybrid

        self.param_count = sum(p.numel() for p in model.parameters())
        self.fireflies = [QuantumFirefly(self.param_count, bounds=(-0.1, 0.1))
                         for _ in range(n_fireflies)]

        self.global_best_position = None
        self.global_best_brightness = float('inf')
        self.criterion = nn.CrossEntropyLoss()
        self.best_fitness_history = []
        self.no_improvement_count = 0

    def evaluate_brightness(self, position):
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
        print("\nQuantum Firefly Algorithm (Classification) Starting...\n")
        print(f"   Fireflies: {self.n_fireflies}")
        print(f"   Iterations: {self.max_iterations}")
        print(f"   Attraction: {self.attraction}")
        print(f"   Randomness: {self.randomness}")
        print(f"   Quantum Factor: {self.quantum_factor}\n")

        print("Evaluating initial fireflies...")
        for firefly in self.fireflies:
            brightness = self.evaluate_brightness(firefly.position)
            firefly.brightness = brightness
            firefly.best_brightness = brightness

            if brightness < self.global_best_brightness:
                self.global_best_brightness = brightness
                self.global_best_position = firefly.position.copy()

        for iteration in range(self.max_iterations):
            adaptive_attraction = self.attraction * (1 - iteration / self.max_iterations)
            adaptive_randomness = self.randomness * (1 - iteration / (2 * self.max_iterations))
            adaptive_quantum = self.quantum_factor * (1 - iteration / self.max_iterations)

            sorted_indices = np.argsort([f.brightness for f in self.fireflies])
            sorted_fireflies = [self.fireflies[i] for i in sorted_indices]

            for i, firefly in enumerate(sorted_fireflies):
                for j in range(i):
                    firefly.attract_to(
                        sorted_fireflies[j],
                        attraction=adaptive_attraction,
                        randomness=adaptive_randomness,
                        quantum_factor=adaptive_quantum,
                        iteration=iteration,
                        max_iterations=self.max_iterations
                    )

                brightness = self.evaluate_brightness(firefly.position)
                firefly.brightness = brightness

                if brightness < firefly.best_brightness:
                    firefly.best_brightness = brightness
                    firefly.best_position = firefly.position.copy()

                if brightness < self.global_best_brightness:
                    self.global_best_brightness = brightness
                    self.global_best_position = firefly.position.copy()
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            self.best_fitness_history.append(self.global_best_brightness)

            if (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                improvement = self.best_fitness_history[0] - self.global_best_brightness
                print(f"Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Loss: {self.global_best_brightness:.6f} | "
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

            val_brightness = self.evaluate_brightness(self._get_weights())
            if val_brightness < self.global_best_brightness:
                self.global_best_brightness = val_brightness
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
    FEATURES_PATH = "./air_pollution_features.npz"

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

    # Convert normalized PM2.5 -> raw PM2.5 -> AQI class labels
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

    model = SimpleModel(X_train.shape[1]).to(device)

    qfo = QuantumFireflyOptimizer(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_fireflies=30,
        max_iterations=150,
        attraction=0.5,
        randomness=0.3,
        quantum_factor=0.1,
        batch_size=32,
        use_hybrid=True
    )

    qfo.optimize()

    print("\nEVALUATION ON TEST SET\n")
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)

    print("=" * 50)
    print("QUANTUM FIREFLY ALGORITHM - CLASSIFICATION RESULTS")
    print("=" * 50)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"],
        zero_division=0
    ))
    print("=" * 50)
