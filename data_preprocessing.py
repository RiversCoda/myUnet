import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BuildingDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, mask_transform=None, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.mode = mode
        self.imgs = os.listdir(img_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            mask_path = os.path.join(self.mask_dir, self.imgs[idx])
            mask = Image.open(mask_path).convert("L")
            if self.mask_transform:
                mask = self.mask_transform(mask)
            return img, mask
        else:
            return img



def get_dataloader(img_dir, mask_dir, batch_size=4, transform=None):
    dataset = BuildingDataset(img_dir, mask_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 定义数据转换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    # 测试数据加载
    train_loader = get_dataloader("train/pic", "train/label", transform=transform)
    for img, mask in train_loader:
        print(img.shape, mask.shape)
        break
