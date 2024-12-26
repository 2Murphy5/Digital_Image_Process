import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import RandomAffine


class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names, transform=None):
        """初始化函数，用于加载图像和对应的掩码的文件
        image_names:图像名称列表
        transform:变换函数"""
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)
        #返回数据集中图像的数量
    def __getitem__(self, idx):
        #根据索引加载图像和对应的掩码，并应用变换
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx].replace('.jpg', '.png'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")#L，单通道图像，灰度值
        #数据增强变换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Directories
image_dir = r"E:\pycharm_pro\Unet\MyUnet\Pocessed_images\CLAHE"
mask_dir = r"E:\Study\DFU\Data\training\training\masks"

# Get all image names
all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Split into training and test sets，测试集占比10%
train_images, test_images = train_test_split(all_images, test_size=0.1, random_state=42)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    # saturation 饱和度
    # hue（色相）是 HSV（Hue, Saturation, Value，即色相、饱和度、明度）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    #添加弹性变换和随机比例抖动
    transforms.ToTensor(),
])
# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create datasets
# 创建训练集和测试集的数据集对象
train_dataset = ImageMaskDataset(image_dir, mask_dir, train_images, train_transform)
test_dataset = ImageMaskDataset(image_dir, mask_dir, test_images, transform)

# Create dataloaders
# 创建数据加载器，用于批量加载数据并打乱训练数据
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        注意力模块：用于调整skip connection中特征图的相关性
        :param F_g: 目标特征图的通道数
        :param F_l: 本地特指图的通道数
        :param F_int: 中间的通道数，用于降低计算的复杂度
        """
        super(AttentionBlock, self).__init__()
        #对目标特征图进行降维
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        #对本地特征图进行降维
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        #计算注意力的权重
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()# sigmoid生成权重
        )
        self.relu = nn.ReLU(inplace=True)# relu激活

    def forward(self, g, x):
        """

        :param g:上采样的特征图
        :param x: 编码器阶段的特征图
        :return: 注意力加权后的特征图
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 合并特征并通过ReLU
        psi = self.relu(g1 + x1)
        #计算注意力的权重
        psi = self.psi(psi)
        #用权重调整原始特征图
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        带注意力机制的UNET模型
        :param in_channels:输入图像的通道数
        :param out_channels: 输出掩码的通道数
        """
        super(AttentionUNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            # Conv-BatchNorm-ReLU模块，用于卷积层定义中
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        # 编码路径的卷积和池化层
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.Conv1 = CBR2d(in_channels, 64)
        self.Conv2 = CBR2d(64, 128)
        self.Conv3 = CBR2d(128, 256)
        self.Conv4 = CBR2d(256, 512)
        self.Conv5 = CBR2d(512, 1024)
        # 解码路径的上采样、注意力模块、卷积
        self.Up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0, bias=True)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = CBR2d(1024, 512)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = CBR2d(512, 256)

        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = CBR2d(256, 128)

        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = CBR2d(128, 64)
        # 最后一层用于生成分割掩码
        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 编码路径，每一层都有Maxpool池化
        e1 = self.Conv1(x)
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5)
        # 解码路径，每一层进行上采样，并使用注意力模块进行skip connection
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, self.Att5(g=d5, x=e4)), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, self.Att4(g=d4, x=e3)), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, self.Att3(g=d3, x=e2)), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, self.Att2(g=d2, x=e1)), dim=1)
        d2 = self.Up_conv2(d2)
        # 最后一层卷积用于生成输出掩码
        out = self.Conv_1x1(d2)

        return out


import torch.optim as optim
from tqdm import tqdm

# Initialize the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionUNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()#二元交叉熵损失，输出的是二分类掩码
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 60
# Add Poly learning rate scheduler
scheduler = PolynomialLR(optimizer, total_iters=len(train_loader) * num_epochs, power=0.9)

# Training loop
num_epochs = 60
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    # 遍历数据集
    for images, masks in tqdm(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()# 梯度清零
        outputs = model(images) #前向传播
        loss = criterion(outputs, masks)# 计算损失
        loss.backward()# 反向传播
        optimizer.step()# 更新模型参数

        train_loss += loss.item()

    # Update the learning rate
    scheduler.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")

    # Evaluate on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
model_save_path = "Unet927.pth"  # 你可以自定义保存路径和文件名
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至 {model_save_path}")
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    for i, (image, mask) in enumerate(test_loader):
        image = image.to(device)
        output = model(image)
        output = torch.sigmoid(output)
        output = output.cpu().numpy()

        # Display the first image in the batch
        plt.subplot(1, 3, 1)
        plt.imshow(image[0].cpu().permute(1, 2, 0))
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Function to perform inference and save masks
def predict_and_save_masks(image_dir, save_dir, model, device, transform, output_size=(640, 480)):
    """
    预测生成掩码图像
    :param image_dir:
    :param save_dir:
    :param model:
    :param device:
    :param transform:图像的处理转换
    :param output_size: 输出的尺寸
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    model.eval()# 评估模式
    with torch.no_grad():
        for image_name in image_names:
            # 加载图像并进行变换
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            # 前向传播，生成输出掩码
            output = model(image)
            output = torch.sigmoid(output).cpu().numpy().squeeze()

            # Convert the output to a binary mask
            mask = (output > 0.5).astype(np.uint8) * 255

            # Resize the mask to the desired output size
            mask_image = Image.fromarray(mask).resize(output_size, Image.Resampling.NEAREST)
            mask_image.save(os.path.join(save_dir, image_name.replace('.jpg', '.png')))

# Directories
inference_image_dir = r"E:\pycharm_pro\Unet\MyUnet\test_images"  # Directory with images for inference
save_mask_dir = r"E:\pycharm_pro\Unet\MyUnet\zhifangtu_masks"  # Directory to save predicted masks

# Define transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionUNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('Unet927.pth'))
model.eval()

# Predict and save masks
predict_and_save_masks(inference_image_dir, save_mask_dir, model, device, transform, output_size=(640, 480))
