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
# QPSO-BASED AIR QUALITY CLASSIFICATION
# ==========================================

import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, accuracy_score)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")
print(f"Device used: {device.type}\n")

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "./data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset"
IND_NEP_PATH = os.path.join(DATASET_PATH, "IND_and_NEP")

# Categories
CATEGORIES = {
    0: 'a_Good',
    1: 'b_Moderate',
    2: 'c_Unhealthy_for_Sensitive_Groups',
    3: 'd_Unhealthy',
    4: 'e_Very_Unhealthy',
    5: 'f_Severe'
}

CATEGORY_NAMES = {
    'a_Good': 'Good',
    'b_Moderate': 'Moderate',
    'c_Unhealthy_for_Sensitive_Groups': 'Unhealthy (Sensitive)',
    'd_Unhealthy': 'Unhealthy',
    'e_Very_Unhealthy': 'Very Unhealthy',
    'f_Severe': 'Severe'
}

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def dark_channel(image, size=15):
    """Extract dark channel prior from image"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return np.mean(dark)

def load_classification_dataset():
    """Load images from category folders"""
    print("="*60)
    print("LOADING CLASSIFICATION DATASET")
    print("="*60)

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
    image_paths = []

    print("\nExtracting features from category folders...")

    for class_id, class_name in CATEGORIES.items():
        class_dir = os.path.join(IND_NEP_PATH, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️  Warning: {class_dir} not found")
            continue

        print(f"\n  Processing {CATEGORY_NAMES[class_name]}...")
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in tqdm(image_files, leave=False):
            img_path = os.path.join(class_dir, img_file)

            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extract DCP
                dcp = dark_channel(image)

                # Extract ResNet features
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(img_tensor).cpu().numpy().flatten()

                features_list.append(feat)
                dcp_list.append(dcp)
                labels_list.append(class_id)
                image_paths.append(img_path)

            except Exception as e:
                continue

        print(f"    Loaded {len([l for l in labels_list if l == class_id])} images")

    X = np.array(features_list)
    DCP = np.array(dcp_list).reshape(-1, 1)
    y = np.array(labels_list)

    # Normalize DCP
    DCP = (DCP - DCP.mean()) / (DCP.std() + 1e-8)

    print(f"\n✅ Dataset loaded!")
    print(f"   Total samples: {len(X)}")
    print(f"   Feature dimensions: {X.shape[1]}")
    for class_id in range(6):
        count = np.sum(y == class_id)
        if count > 0:
            print(f"   {CATEGORY_NAMES[CATEGORIES[class_id]]}: {count} samples")

    return X, DCP, y

# ==========================================
# CLASSIFICATION MODEL
# ==========================================

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# QUANTUM SWARM OPTIMIZATION FOR CLASSIFICATION
# ==========================================

class QuantumParticle:
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(bounds[0] * 0.1, bounds[1] * 0.1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')  # Higher is better (accuracy)
        self.bounds = bounds

    def update_quantum(self, global_best, w=0.9, c1=1.7, c2=1.7, quantum_factor=0.05):
        r1 = np.random.rand(*self.position.shape)
        r2 = np.random.rand(*self.position.shape)

        self.velocity = (w * self.velocity +
                        c1 * r1 * (self.best_position - self.position) +
                        c2 * r2 * (global_best - self.position))

        max_velocity = (self.bounds[1] - self.bounds[0]) * 0.2
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

        if np.random.rand() < quantum_factor:
            self.position = np.random.uniform(self.bounds[0], self.bounds[1],
                                            self.position.shape)
        else:
            self.position = self.position + self.velocity

        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])

class QuantumSwarmClassifier:
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 n_particles=40, max_iterations=100):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.param_count = sum(p.numel() for p in model.parameters())
        self.particles = [QuantumParticle(self.param_count, bounds=(-0.1, 0.1))
                         for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('-inf')

    def evaluate_fitness(self, position):
        """Evaluate model accuracy with given weights"""
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
            with torch.no_grad():
                X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(self.y_val, dtype=torch.long)

                outputs = self.model(X_val_tensor)
                predictions = torch.argmax(outputs, dim=1).numpy()
                accuracy = accuracy_score(self.y_val, predictions)

            return accuracy
        except:
            return 0.0

    def optimize(self, verbose=True):
        """Run quantum swarm optimization"""
        print("\n🌀 Quantum Swarm Classification Optimization\n")

        # Initial evaluation
        print("Evaluating initial particles...")
        for particle in self.particles:
            fitness = self.evaluate_fitness(particle.position)
            particle.best_fitness = fitness

            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

        # Optimization loop
        for iteration in range(self.max_iterations):
            w = 0.9 - (0.5 * iteration / self.max_iterations)

            for particle in self.particles:
                particle.update_quantum(self.global_best_position, w=w, c1=1.7, c2=1.7,
                                       quantum_factor=0.05)

                fitness = self.evaluate_fitness(particle.position)

                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            if verbose and (iteration + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"  Iteration {iteration+1:3d}/{self.max_iterations} | "
                      f"Best Accuracy: {self.global_best_fitness:.4f}")

        # Hybrid refinement with gradient descent
        print("\n🔧 Hybrid Refinement (Gradient Descent)...")
        self._hybrid_refinement()

        # Set best weights
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = torch.tensor(
                self.global_best_position[idx:idx+param_size].reshape(param.shape),
                dtype=param.dtype
            )
            idx += param_size

        print("✅ Optimization Complete!\n")

    def _get_weights(self):
        """Extract weights as flat vector"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def _hybrid_refinement(self, epochs=15):
        """Fine-tune with gradient descent"""
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
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            loss_total = 0

            for i in range(0, len(self.X_train), 32):
                X_batch = torch.tensor(self.X_train[i:i+32], dtype=torch.float32)
                y_batch = torch.tensor(self.y_train[i:i+32], dtype=torch.long)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                loss_total += loss.item()

            # Validate
            self.model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(self.y_val, dtype=torch.long)

                outputs = self.model(X_val_tensor)
                predictions = torch.argmax(outputs, dim=1).numpy()
                val_accuracy = accuracy_score(self.y_val, predictions)

                if val_accuracy > self.global_best_fitness:
                    self.global_best_fitness = val_accuracy
                    self.global_best_position = self._get_weights().copy()

# ==========================================
# EVALUATION
# ==========================================

def evaluate_classifier(model, X_test, y_test):
    """Evaluate classification performance"""
    print("\n" + "="*60)
    print("CLASSIFICATION EVALUATION")
    print("="*60 + "\n")

    model.eval()
    y_pred = []

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        outputs = model(X_test_tensor)
        y_pred = torch.argmax(outputs, dim=1).numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Overall Accuracy:        {accuracy:.4f}")
    print(f"Precision (Macro):       {precision_macro:.4f}")
    print(f"Precision (Weighted):    {precision_weighted:.4f}")
    print(f"Recall (Macro):          {recall_macro:.4f}")
    print(f"F1 Score (Macro):        {f1_macro:.4f}")

    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS PRECISION")
    print("="*60 + "\n")

    precisions = precision_score(y_test, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)

    for class_id in range(6):
        class_name = CATEGORY_NAMES[CATEGORIES[class_id]]
        print(f"{class_name:30s}: Precision={precisions[class_id]:.4f}, "
              f"Recall={recalls[class_id]:.4f}")

    # Confusion matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60 + "\n")

    cm = confusion_matrix(y_test, y_pred)
    print("Predicted →")
    print("Actual ↓")

    # Header
    print("     ", end="")
    for i in range(6):
        print(f"{i:8d}", end="")
    print()

    # Rows
    for i in range(6):
        print(f"{i}:   ", end="")
        for j in range(6):
            print(f"{cm[i][j]:8d}", end="")
        print()

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'f1_score': f1_macro,
        'per_class_precision': precisions
    }

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("QPSO-BASED AIR QUALITY CLASSIFICATION")
    print("="*60 + "\n")

    start_time = time.time()

    # Load dataset
    X, DCP, y = load_classification_dataset()
    X_combined = np.concatenate([X, DCP], axis=1)

    # Split data: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Initialize model
    model = ClassificationModel(X_train.shape[1], num_classes=6).to(device)

    # Run QPSO optimization
    qso = QuantumSwarmClassifier(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_particles=40,
        max_iterations=100
    )

    qso.optimize(verbose=True)

    # Evaluate
    results = evaluate_classifier(model, X_test, y_test)

    # Summary
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS")
    print("="*60)
    print(f"Overall Accuracy:    {results['accuracy']:.4f}")
    print(f"Macro Precision:     {results['precision_macro']:.4f}")
    print(f"Weighted Precision:  {results['precision_weighted']:.4f}")
    print(f"Macro Recall:        {results['recall_macro']:.4f}")
    print(f"Macro F1 Score:      {results['f1_score']:.4f}")
    print("="*60)

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed:.2f}s")
