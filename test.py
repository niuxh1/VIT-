import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import VisionTransformer  # 确保 model.py 与 test.py 在同一目录下
from tqdm import tqdm  # 用于显示进度条

# 设置超参数
img_size = 224
batch_size = 128
num_classes = 100  # CIFAR-100 有100个类

# 判断是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 下载并加载测试集
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型并加载预训练权重
model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('vit_model_cifar100.pth', map_location=device))  # 加载训练好的模型
model.eval()  # 设置模型为评估模式

# 计算测试集上的准确率
correct = 0
total = 0

# tqdm 实时显示测试进度
with torch.no_grad():
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # 前向传播
        _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条上的准确率
        progress_bar.set_postfix(accuracy=100 * correct / total)

# 输出最终准确率
accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

# 可视化部分测试结果
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 获取部分测试样本和预测结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 显示图像
imshow(torchvision.utils.make_grid(images.cpu()[:8]))

# 输出预测结果
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 打印真实标签和预测标签
classes = test_dataset.classes
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]}' for j in range(8)))
print('Predicted: ', ' '.join(f'{classes[predicted[j]]}' for j in range(8)))
