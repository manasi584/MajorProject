# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "opencv-python>=4.13.0.92",
#     "pandas>=2.3.3",
#     "torch>=2.8.0",
#     "torchvision>=0.23.0",
#     "tqdm>=4.67.3",
#     "scikit-learn>=1.6.1",
#     "scipy>=1.13.1",
# ]
# ///

# ==========================================
# CNN ONLY - FEATURE EXTRACTION + TRAINING
# ==========================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

device = torch.device("cpu")
print(f"Device used: {device.type}")

# ==========================================
# FEATURE EXTRACTION
# ==========================================

print("\n🚀 Starting Feature Extraction (CNN Only)\n")

# -------- PATHS --------
DATASET_PATH = "./data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset"
CSV_PATH = os.path.join(DATASET_PATH, "IND_and_Nep_AQI_Dataset.csv")
IMAGE_DIR = os.path.join(DATASET_PATH, "All_img")

SAVE_PATH = "./air_pollution_features_cnn_only.npz"

# -------- LOAD DATA --------
df = pd.read_csv(CSV_PATH)

PM25_MAX = df['PM2.5'].max()
df['PM2.5'] = df['PM2.5'] / PM25_MAX

# -------- MODEL --------
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
labels_list = []

print("Extracting CNN features...")

for img_name, label in tqdm(zip(df['Filename'], df['PM2.5']), total=len(df)):
    img_path = os.path.join(IMAGE_DIR, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transform
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(img_tensor).cpu().numpy().flatten()

    features_list.append(feat)
    labels_list.append(label)

# Convert to arrays
X = np.array(features_list)
y = np.array(labels_list)

# Save features
np.savez(SAVE_PATH, X=X, y=y, PM25_MAX=PM25_MAX)

print(f"\n✅ Features saved at: {SAVE_PATH}")
print(f"Feature shape: {X.shape}")

# ==========================================
# TRAINING
# ==========================================

print("\n🚀 Starting Training (CNN Only Features)\n")

# -------- LOAD FEATURES --------
data = np.load(SAVE_PATH)

X = data['X']
y = data['y']
PM25_MAX = data['PM25_MAX']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to tensors
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

# -------- MODEL --------
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

model = SimpleModel(X_train.shape[1]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# -------- TRAIN --------
EPOCHS = 20

print("Training Started\n")

for epoch in range(EPOCHS):
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

    print(f"Epoch {epoch+1}: Loss = {loss_total / len(train_loader):.5f}")


# -------- EVALUATION --------
y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch).squeeze().cpu().numpy()
        y_pred.extend(outputs)
        y_true.extend(y_batch.numpy())

y_true = np.array(y_true) * PM25_MAX
y_pred = np.array(y_pred) * PM25_MAX

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
pearson_corr, _ = pearsonr(y_true, y_pred)

print("\n📊 FINAL RESULTS (CNN ONLY)")
print("="*30)
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2: {r2:.3f}")
print(f"Pearson: {pearson_corr:.3f}")
print("="*30)
