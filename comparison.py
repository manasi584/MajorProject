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
# COMPREHENSIVE COMPARISON OF ALL 3 APPROACHES
# ==========================================

import os
import cv2
import time
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
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "./data/Air Pollution Image Dataset/Combined_Dataset"
CSV_PATH = os.path.join(DATASET_PATH, "IND_and_Nep_AQI_Dataset.csv")
IMAGE_DIR = os.path.join(DATASET_PATH, "All_img")
FEATURES_PATH = "./air_pollution_features.npz"

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def dark_channel(image, size=15):
    """Extract dark channel prior from image"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return np.mean(dark)

def extract_features(use_dcp=True):
    """Extract ResNet18 features and optionally Dark Channel Prior"""
    print("\n" + "="*60)
    print("FEATURE EXTRACTION")
    print("="*60)

    df = pd.read_csv(CSV_PATH)
    PM25_MAX = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / PM25_MAX

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
        img_path = os.path.join(IMAGE_DIR, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if use_dcp:
            dcp = dark_channel(image)
            dcp_list.append(dcp)

        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().flatten()

        features_list.append(feat)
        labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    if use_dcp:
        DCP = np.array(dcp_list).reshape(-1, 1)
        DCP = (DCP - DCP.mean()) / (DCP.std() + 1e-8)
        return X, DCP, y, PM25_MAX
    else:
        return X, None, y, PM25_MAX

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
# APPROACH 1: CNN-ONLY
# ==========================================

def train_cnn_only(X, y, PM25_MAX):
    """Train using CNN features only (80/20 split, no validation)"""
    print("\n" + "="*60)
    print("APPROACH 1: CNN-ONLY (ResNet18 Features)")
    print("="*60)

    start_time = time.time()

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train
    model = SimpleModel(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("\nTraining (20 epochs)...")
    for epoch in range(20):
        model.train()
        loss_total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss_total / len(train_loader):.5f}")

    # Evaluate
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch).squeeze().cpu().numpy()
            y_pred.extend(outputs)

    y_true = y_test * PM25_MAX
    y_pred = np.array(y_pred) * PM25_MAX

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)

    elapsed = time.time() - start_time

    return {
        'name': 'CNN-Only',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'time': elapsed,
        'features': 'ResNet18 only',
        'split': '80/20'
    }

# ==========================================
# APPROACH 2: END-TO-END
# ==========================================

def train_end_to_end(X, DCP, y, PM25_MAX):
    """Train with gradient descent, ResNet18 + DCP, 60/20/20 split"""
    print("\n" + "="*60)
    print("APPROACH 2: END-TO-END (Gradient Descent)")
    print("="*60)

    start_time = time.time()

    X_combined = np.concatenate([X, DCP], axis=1)

    # Split: 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Train
    model = SimpleModel(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("\nTraining (20 epochs)...")
    for epoch in range(20):
        model.train()
        loss_total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

        # Validation
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch).squeeze()
                val_loss = criterion(outputs, y_batch)
                val_loss_total += val_loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Train = {loss_total / len(train_loader):.5f}, "
                  f"Val = {val_loss_total / len(val_loader):.5f}")

    # Evaluate
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch).squeeze().cpu().numpy()
            y_pred.extend(outputs)

    y_true = y_test * PM25_MAX
    y_pred = np.array(y_pred) * PM25_MAX

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)

    elapsed = time.time() - start_time

    return {
        'name': 'End-to-End',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'time': elapsed,
        'features': 'ResNet18 + DCP',
        'split': '60/20/20'
    }

# ==========================================
# APPROACH 3: QUANTUM SWARM OPTIMIZATION
# ==========================================

class QuantumParticle:
    def __init__(self, dim, bounds=(-0.1, 0.1)):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(bounds[0] * 0.1, bounds[1] * 0.1, dim)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
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

class QuantumSwarmOptimizer:
    def __init__(self, model, X_train, y_train, X_val, y_val, n_particles=40, max_iterations=150):
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
        self.global_best_fitness = float('inf')
        self.criterion = nn.MSELoss()

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
            with torch.no_grad():
                X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(self.y_val, dtype=torch.float32)
                outputs = self.model(X_val_tensor).squeeze()
                loss = self.criterion(outputs, y_val_tensor)
                total_loss = loss.item()
            return total_loss
        except:
            return float('inf')

    def optimize(self, verbose=False):
        for i, particle in enumerate(self.particles):
            fitness = self.evaluate_fitness(particle.position)
            particle.best_fitness = fitness
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()

        for iteration in range(self.max_iterations):
            w = 0.9 - (0.5 * iteration / self.max_iterations)

            for particle in self.particles:
                particle.update_quantum(self.global_best_position, w=w, c1=1.7, c2=1.7, quantum_factor=0.05)
                fitness = self.evaluate_fitness(particle.position)

                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()

            if verbose and (iteration + 1) % 30 == 0:
                print(f"  Iteration {iteration+1}: Best Loss = {self.global_best_fitness:.6f}")

        # Hybrid refinement
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

        for i in range(10):
            self.model.train()
            for j in range(0, len(self.X_train), 32):
                X_batch = torch.tensor(self.X_train[j:j+32], dtype=torch.float32)
                y_batch = torch.tensor(self.y_train[j:j+32], dtype=torch.float32)

                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

def train_quantum_swarm(X, DCP, y, PM25_MAX):
    """Train with QPSO + hybrid gradient descent"""
    print("\n" + "="*60)
    print("APPROACH 3: QUANTUM SWARM OPTIMIZATION (QPSO)")
    print("="*60)

    start_time = time.time()

    X_combined = np.concatenate([X, DCP], axis=1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = SimpleModel(X_train.shape[1]).to(device)

    print("\nOptimizing with Quantum Swarm...")
    qso = QuantumSwarmOptimizer(model, X_train, y_train, X_val, y_val, n_particles=40, max_iterations=150)
    qso.optimize(verbose=True)

    print("Fine-tuning with gradient descent...")
    qso.optimize(verbose=False)

    # Evaluate
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch).squeeze().cpu().numpy()
            y_pred.extend(outputs)

    y_true = y_test * PM25_MAX
    y_pred = np.array(y_pred) * PM25_MAX

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)

    elapsed = time.time() - start_time

    return {
        'name': 'QPSO + Hybrid',
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson,
        'time': elapsed,
        'features': 'ResNet18 + DCP',
        'split': '60/20/20'
    }

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON: ALL 3 APPROACHES")
    print("="*60)

    # Load or extract features
    if os.path.exists(FEATURES_PATH):
        print("\n📂 Loading pre-extracted features...")
        data = np.load(FEATURES_PATH)
        X = data['X']
        DCP = data['DCP']
        y = data['y']
        PM25_MAX = data['PM25_MAX']
    else:
        print("\n🔨 Extracting features...")
        X, DCP, y, PM25_MAX = extract_features(use_dcp=True)
        np.savez(FEATURES_PATH, X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)

    # Extract CNN-only features
    X_cnn_only = X  # Same as X

    results = []

    # Run Approach 1: CNN-Only
    try:
        result1 = train_cnn_only(X_cnn_only, y, PM25_MAX)
        results.append(result1)
    except Exception as e:
        print(f"❌ CNN-Only failed: {e}")

    # Run Approach 2: End-to-End
    try:
        result2 = train_end_to_end(X, DCP, y, PM25_MAX)
        results.append(result2)
    except Exception as e:
        print(f"❌ End-to-End failed: {e}")

    # Run Approach 3: QPSO
    try:
        result3 = train_quantum_swarm(X, DCP, y, PM25_MAX)
        results.append(result3)
    except Exception as e:
        print(f"❌ QPSO failed: {e}")

    # Print comparison table
    print("\n" + "="*100)
    print("FINAL COMPARISON RESULTS")
    print("="*100)

    comparison_df = pd.DataFrame(results)

    # Reorder columns for better display
    display_cols = ['name', 'mae', 'rmse', 'r2', 'pearson', 'time', 'features', 'split']
    print("\n" + comparison_df[display_cols].to_string(index=False))

    # Highlight best results
    print("\n" + "="*100)
    print("BEST RESULTS:")
    print("="*100)
    print(f"🏆 Best MAE:         {comparison_df.loc[comparison_df['mae'].idxmin(), 'name']:20s} ({comparison_df['mae'].min():.3f})")
    print(f"🏆 Best RMSE:        {comparison_df.loc[comparison_df['rmse'].idxmin(), 'name']:20s} ({comparison_df['rmse'].min():.3f})")
    print(f"🏆 Best R²:          {comparison_df.loc[comparison_df['r2'].idxmax(), 'name']:20s} ({comparison_df['r2'].max():.3f})")
    print(f"🏆 Best Pearson:     {comparison_df.loc[comparison_df['pearson'].idxmax(), 'name']:20s} ({comparison_df['pearson'].max():.3f})")
    print(f"⚡ Fastest:          {comparison_df.loc[comparison_df['time'].idxmin(), 'name']:20s} ({comparison_df['time'].min():.2f}s)")

    print("\n" + "="*100)
    print("✨ Comparison complete!")
    print("="*100)
