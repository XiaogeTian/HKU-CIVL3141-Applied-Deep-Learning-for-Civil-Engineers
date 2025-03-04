import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 数据集类
class PotholeDataset(Dataset):
    def __init__(self, pothole_dir, normal_dir, transform=None):
        self.pothole_dir = pothole_dir
        self.normal_dir = normal_dir
        self.transform = transform

        if not os.path.exists(pothole_dir):
            raise ValueError(f"Pothole directory does not exist: {pothole_dir}")
        if not os.path.exists(normal_dir):
            raise ValueError(f"Normal directory does not exist: {normal_dir}")

        self.pothole_images = [os.path.join(pothole_dir, f) for f in os.listdir(pothole_dir) if f.endswith('.jpg')]
        self.normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.jpg')]
        self.images = self.pothole_images + self.normal_images
        self.labels = [1] * len(self.pothole_images) + [0] * len(self.normal_images)

        # 检查是否有图像加载
        if not self.images:
            raise ValueError("No valid .jpg images found in the directories!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (256, 256))  # 调整图像大小

        if label == 1:  # 有坑洞
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img_gray, 50, 150)
            edges_resized = cv2.resize(edges, (256, 256), interpolation=cv2.INTER_NEAREST)  # 调整边缘检测结果
            pseudo_mask = cv2.dilate(edges_resized, np.ones((5, 5), np.uint8), iterations=1)
            pseudo_mask = (pseudo_mask > 0).astype(np.uint8)
        else:  # 无坑洞
            pseudo_mask = np.zeros((256, 256), dtype=np.uint8)

        if self.transform:
            image_resized = self.transform(image_resized)

        return image_resized, torch.from_numpy(pseudo_mask).long()



# U-Net 模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):  # 2类：背景+坑洞
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.up2(b)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)


# 数据准备
pothole_dir = "./data/pothole/hole"
normal_dir = "./data/pothole/normal"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = PotholeDataset(pothole_dir, normal_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# 模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, pseudo_masks in train_loader:
        images, pseudo_masks = images.to(device), pseudo_masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, pseudo_masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")