# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2.0.2",
#     "opencv-python>=4.13.0.92",
#     "pandas>=2.3.3",
#     "torch>=2.8.0",
#     "torchvision>=0.23.0",
#     "tqdm>=4.67.3",
# ]
# ///

# ==========================================
# FEATURE EXTRACTION - RESNET50 + DCP
# ==========================================

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DARK CHANNEL PRIOR
# ==========================================

def dark_channel(image, size=15):
    """Compute dark channel prior from image"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return np.mean(dark)

# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features_resnet50(dataset_path, csv_path, image_dir, pm25_max=None):
    """Extract ResNet50 features + Dark Channel Prior"""
    print("\n" + "="*60)
    print("FEATURE EXTRACTION - RESNET50 + DCP")
    print("="*60)

    # Load dataset metadata
    print("\n1. Loading dataset metadata...")
    df = pd.read_csv(csv_path)
    print(f"   Total samples: {len(df)}")

    if pm25_max is None:
        pm25_max = df['PM2.5'].max()
    df['PM2.5'] = df['PM2.5'] / pm25_max
    print(f"   PM2.5 max: {pm25_max:.2f}")

    # Load ResNet50 backbone
    print("\n2. Loading ResNet50 pre-trained model...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()  # Remove classification head
    model = model.to(device)
    model.eval()
    print("   ResNet50 loaded (feature extraction mode)")

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Feature extraction
    print("\n3. Extracting features...")
    features_list = []
    dcp_list = []
    labels_list = []

    for img_name, label in tqdm(zip(df['Filename'], df['PM2.5']), total=len(df), desc="Extracting"):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Extract dark channel prior
        dcp = dark_channel(image)

        # Extract ResNet50 features
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).cpu().numpy().flatten()

        features_list.append(feat)
        dcp_list.append(dcp)
        labels_list.append(label)

    # Aggregate results
    X = np.array(features_list)
    DCP = np.array(dcp_list).reshape(-1, 1)
    y = np.array(labels_list)

    # Normalize DCP
    DCP = (DCP - DCP.mean()) / (DCP.std() + 1e-8)

    print(f"\n4. Feature extraction complete!")
    print(f"   ResNet50 features shape: {X.shape}")
    print(f"   DCP features shape: {DCP.shape}")
    print(f"   Labels shape: {y.shape}")

    return X, DCP, y, pm25_max

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    DATASET_PATH = "./data/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset"
    CSV_PATH = os.path.join(DATASET_PATH, "IND_and_Nep_AQI_Dataset.csv")
    IMAGE_DIR = os.path.join(DATASET_PATH, "All_img")
    OUTPUT_PATH = "./features_resnet50.npz"

    # Extract features
    X, DCP, y, PM25_MAX = extract_features_resnet50(DATASET_PATH, CSV_PATH, IMAGE_DIR)

    # Save
    print(f"\n5. Saving features to {OUTPUT_PATH}...")
    np.savez(OUTPUT_PATH, X=X, DCP=DCP, y=y, PM25_MAX=PM25_MAX)
    print("   ✓ Features saved!")
    print("\n" + "="*60)
