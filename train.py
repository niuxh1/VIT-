import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from model import VisionTransformer  # 确保 model.py 与 train.py 在同一目录下
from tqdm import tqdm  # 用于显示进度条

# 设置超参数
img_size = 224
patch_size = 16
embed_dim = 512
depth = 8
num_heads = 8
num_classes = 100  # CIFAR-100 有100个类
batch_size = 128
num_epochs = 20  # 增加训练轮数
learning_rate = 0.001

# 判断是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),  # 调整为 224x224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(img_size, padding=4),  # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 下载并加载训练集
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, num_classes=num_classes).to(device)
criterion = F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch [{epoch+1}/{num_epochs}]')

    for i, (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

        # 更新进度条
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 保存模型
torch.save(model.state_dict(), 'vit_model_cifar100.pth')
print('Model saved as vit_model_cifar100.pth')
