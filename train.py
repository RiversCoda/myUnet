import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 引入进度条库
from unet_model import UNet
from data_preprocessing import BuildingDataset

# 超参数设置
epochs = 8
batch_size = 8
learning_rate = 0.01

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 为掩码定义一个单独的变换序列
mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

# 在创建 BuildingDataset 实例时传入 mask_transform 参数
train_loader = DataLoader(
    BuildingDataset("train/pic", "train/label", transform=transform, mask_transform=mask_transform),
    batch_size=batch_size, shuffle=True)

# 模型初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
    for i, (images, masks) in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix({"Loss": running_loss / ((i + 1) * batch_size)})

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"\nEpoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "unet_model2.pth")
