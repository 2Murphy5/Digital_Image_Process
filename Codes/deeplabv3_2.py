import os
import random
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
INPUT_HEIGHT = 480
INPUT_WIDTH = 640
NUM_CLASSES = 2  # Adjust for your dataset
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Dataset Class
class MedicalSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask) * (NUM_CLASSES - 1)  # Normalize mask values

        return image, mask.squeeze(0).long()  # Long type for CrossEntropyLoss


# 2. Define Model
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.model = deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)


# 3. Data Preparation
def prepare_data(input_folder, mask_folder, split_ratio=0.8, input_format="jpg", mask_format="png"):
    image_paths = sorted(glob(os.path.join(input_folder, f"*.{input_format}")))
    mask_paths = sorted(glob(os.path.join(mask_folder, f"*.{mask_format}")))

    assert len(image_paths) == len(mask_paths), "Image and mask counts do not match!"

    data = list(zip(image_paths, mask_paths))
    random.shuffle(data)
    train_size = int(len(data) * split_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:]

    train_images, train_masks = zip(*train_data)
    val_images, val_masks = zip(*val_data)

    transform = transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = MedicalSegmentationDataset(train_images, train_masks, transform)
    val_dataset = MedicalSegmentationDataset(val_images, val_masks, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


# Dice Loss Calculation
def dice_loss(pred, target):
    smooth = 1e-6
    intersection = torch.sum(pred * target)
    dice = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return 1 - dice


# 4. Train and Evaluate
def train_model(model, train_loader, val_loader, epochs, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    scaler = GradScaler()  # For mixed-precision training

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with autocast():  # Mixed precision
                outputs = model(images)['out']
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                with autocast():  # Mixed precision
                    outputs = model(images)['out']
                    loss = criterion(outputs, masks)
                    pred_mask = torch.argmax(outputs, dim=1)

                    # Dice coefficient
                    dice = 1 - dice_loss(pred_mask.float(), masks.float())

                    # IoU
                    intersection = torch.sum((pred_mask == 1) & (masks == 1))
                    union = torch.sum((pred_mask == 1) | (masks == 1))
                    iou = intersection / (union + 1e-6)

                val_loss += loss.item()
                val_dice += dice.item()
                val_iou += iou.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # Save the model
        torch.save(model.state_dict(), f"{save_path}_epoch_{epoch + 1}.pth")


# Main script
if __name__ == "__main__":
    input_folder = r"E:\pycharm_pro\Unet\MyUnet\train_images"  # Replace with your image folder
    mask_folder = r"E:\pycharm_pro\Unet\MyUnet\train_masks"  # Replace with your mask folder
    model_save_path = r"E:\pycharm_pro\Unet\MyUnet\DeeplabV3plus\deeplabv3plus_model2"  # Base path for saving models

    # Prepare data
    train_loader, val_loader = prepare_data(input_folder, mask_folder)

    # Define and initialize the model
    model = DeepLabV3Plus(NUM_CLASSES).to(DEVICE)

    # Train the model
    train_model(model, train_loader, val_loader, EPOCHS, model_save_path)

    print("Training completed and model saved.")
