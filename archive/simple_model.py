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
# END-TO-END PIPELINE
# Feature Extraction + Training + Evaluation
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
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cpu")
print(f"Device used: {device.type}\n")

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_PATH = "./data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset"
CSV_PATH = os.path.join(DATASET_PATH, "IND_and_Nep_AQI_Dataset.csv")
IMAGE_DIR = os.path.join(DATASET_PATH, "All_img")
FEATURES_PATH = "./air_pollution_features.npz"

EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def dark_channel(image, size=15):
    """Extract dark channel prior from image"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return np.mean(dark)

def extract_features():
    """Extract ResNet18 + Dark Channel Prior features from images"""
    print("=" * 50)
    print("FEATURE EXTRACTION")
    print("=" * 50)

    # Load dataset
    print("\n📊 Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    PM25_MAX = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / PM25_MAX

    # Initialize ResNet18 model
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

    # Extract features
    print("🚀 Extracting features...")
    for img_name, label in tqdm(zip(df['Filename'], df['PM2.5']), total=len(df)):
        img_path = os.path.join(IMAGE_DIR, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dark Channel Prior
        dcp = dark_channel(image)

        # ResNet18 features
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().flatten()

        features_list.append(feat)
        dcp_list.append(dcp)
        labels_list.append(label)

    # Convert to arrays
    X = np.array(features_list)
    DCP = np.array(dcp_list).reshape(-1, 1)
    y = np.array(labels_list)

    # Normalize DCP
    DCP = (DCP - DCP.mean()) / (DCP.std() + 1e-8)

    print(f"\n✅ Features extracted!")
    print(f"   X shape: {X.shape}")
    print(f"   DCP shape: {DCP.shape}")
    print(f"   y shape: {y.shape}")

    return X, DCP, y, PM25_MAX

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
# TRAINING
# ==========================================

def train_model(model, train_loader, val_loader, epochs, learning_rate):
    """Train the model with validation monitoring"""
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50 + "\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss_total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        # Validation phase
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch).squeeze()
                val_loss = criterion(outputs, y_batch)
                val_loss_total += val_loss.item()

        train_loss_avg = train_loss_total / len(train_loader)
        val_loss_avg = val_loss_total / len(val_loader)

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss_avg:.5f} | "
              f"Val Loss: {val_loss_avg:.5f}")

    print("\n✅ Training complete!")

# ==========================================
# EVALUATION
# ==========================================

def evaluate_model(model, test_loader, PM25_MAX):
    """Evaluate model on test set"""
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50 + "\n")

    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch).squeeze().cpu().numpy()
            y_pred.extend(outputs)
            y_true.extend(y_batch.numpy())

    # Denormalize
    y_true = np.array(y_true) * PM25_MAX
    y_pred = np.array(y_pred) * PM25_MAX

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print("📊 FINAL RESULTS")
    print("=" * 50)
    print(f"MAE (Mean Absolute Error):  {mae:.3f}")
    print(f"RMSE (Root Mean Squared):   {rmse:.3f}")
    print(f"R² Score:                   {r2:.3f}")
    print(f"Pearson Correlation:        {pearson_corr:.3f}")
    print("=" * 50)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'pearson': pearson_corr
    }

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Step 1: Feature Extraction
    if os.path.exists(FEATURES_PATH):
        print("📂 Loading pre-extracted features...\n")
        data = np.load(FEATURES_PATH)
        X = data['X']
        DCP = data['DCP']
        y = data['y']
        PM25_MAX = data['PM25_MAX']
    else:
        X, DCP, y, PM25_MAX = extract_features()
        print(f"\n💾 Saving features to {FEATURES_PATH}...")
        np.savez(FEATURES_PATH, X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)

    # Step 2: Data Preparation
    print("\n" + "=" * 50)
    print("DATA PREPARATION")
    print("=" * 50 + "\n")

    # Combine features
    X_combined = np.concatenate([X, DCP], axis=1)
    print(f"Combined features shape: {X_combined.shape}")

    # Split: 60% train, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_combined, y, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Val set:   {X_val.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Create data loaders
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Step 3: Model Training
    model = SimpleModel(X_train.shape[1]).to(device)
    train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE)

    # Step 4: Evaluation
    results = evaluate_model(model, test_loader, PM25_MAX)

    print("\n✨ Pipeline complete!")


#commands 
# uv add --script <file_name> opencv-python torchvision