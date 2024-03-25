import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from unet_model import UNet
from data_preprocessing import BuildingDataset
import numpy as np

# 定义精确度计算函数
def calculate_accuracy(pred_dir, label_dir):
    pred_files = sorted(os.listdir(pred_dir))
    label_files = sorted(os.listdir(label_dir))
    total_accuracy = 0
    for pred_file, label_file in zip(pred_files, label_files):
        pred = np.array(Image.open(os.path.join(pred_dir, pred_file)))
        label = np.array(Image.open(os.path.join(label_dir, label_file)))
        accuracy = np.sum(pred == label) / pred.size
        total_accuracy += accuracy
    return total_accuracy / len(pred_files)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load("unet_model2.pth", map_location=device))
model = model.to(device)
model.eval()

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_loader = DataLoader(
    BuildingDataset("test/pic", transform=transform, mode='test'),
    batch_size=1, shuffle=False)

# 创建输出文件夹
sigmoid_output_dir = "test/output2/sigmoid"
bin_output_dir = "test/output2/bin"
os.makedirs(sigmoid_output_dir, exist_ok=True)
os.makedirs(bin_output_dir, exist_ok=True)

# 预测并保存结果
with torch.no_grad():
    for i, images in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        sigmoid_preds = torch.sigmoid(outputs).cpu().numpy()

        # 保存sigmoid结果
        sigmoid_img = (sigmoid_preds[0, 0] * 255).astype("uint8")
        Image.fromarray(sigmoid_img).save(os.path.join(sigmoid_output_dir, f"{i+1}.tif"))

        # 保存二值化结果
        bin_preds = (sigmoid_preds > 0.25).astype(float)
        bin_img = (bin_preds[0, 0] * 255).astype("uint8")
        Image.fromarray(bin_img).save(os.path.join(bin_output_dir, f"{i+1}.tif"))

print("Prediction completed and saved.")

# 计算精确度
accuracy = calculate_accuracy(bin_output_dir, "test/label")
print(f"Accuracy: {accuracy:.4f}")
