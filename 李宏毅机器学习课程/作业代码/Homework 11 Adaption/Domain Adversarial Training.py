"""
source domain: photo
target domain: sketch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
import os

torch.manual_seed(42)

# 设置模型保存路径
model_dir = './pretrained_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.hub.set_dir(model_dir)

# 梯度反转层 (GRL)
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# DANN 网络模型

class DANN(nn.Module):
    def __init__(self, num_classes=7):
        super(DANN, self).__init__()

        # 特征提取器 (使用预训练的 ResNet18)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = resnet.fc.in_features

        # 标签分类器
        self.class_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes)
        )

        # 领域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 2)  # 二分类：0源域，1目标域
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # 调用梯度反转
        reverse_features = GradientReversalLayer.apply(features, alpha)

        class_output = self.class_classifier(features)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output

# 超参数与环境配置
source_dir = './PCAS_Datasets/photo'
target_dir = './PCAS_Datasets/sketch'
batch_size = 32
num_epochs = 20
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# 数据预处理与加载

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

source_dataset = datasets.ImageFolder(root=source_dir, transform=transform)
target_dataset = datasets.ImageFolder(root=target_dir, transform=transform)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# 初始化模型与优化器
model = DANN(num_classes=7).to(device)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# 6. 开始训练循环
best_accuracy = 0
len_dataloader = min(len(source_loader), len(target_loader))

for epoch in range(num_epochs):
    model.train()
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    total_class_loss = 0.0
    total_domain_loss = 0.0

    for i in range(len_dataloader):
        # 计算动态 alpha
        p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 获取数据并送入设备
        source_data, source_label = next(source_iter)
        source_data, source_label = source_data.to(device), source_label.to(device)
        domain_label_source = torch.zeros(batch_size, 2).to(device)  # 源域标签 0
        domain_label_source[:, 0] = 1

        target_data, _ = next(target_iter)
        target_data = target_data.to(device)
        domain_label_target = torch.ones(batch_size, 2).to(device)  # 目标域标签 1
        domain_label_target[:, 1] = 1

        # 清空梯度
        optimizer.zero_grad()

        # 源域前向传播
        class_output_source, domain_output_source = model(source_data, alpha)
        loss_source_label = class_criterion(class_output_source, source_label)
        loss_source_domain = domain_criterion(domain_output_source, domain_label_source)

        # 目标域前向传播
        # 不需要目标域数据的分类结果和损失，否则就相当于把原始域和目标域的数据先后送进模型，并没有实现领域自适应的效果。
        _, domain_output_target = model(target_data, alpha)
        loss_target_domain = domain_criterion(domain_output_target, domain_label_target)

        # 综合损失并反向传播
        loss = loss_source_label + loss_source_domain + loss_target_domain
        loss.backward()
        optimizer.step()

        # 记录日志
        total_class_loss += loss_source_label.item()
        total_domain_loss += (loss_source_domain.item() + loss_target_domain.item())

    # 每一轮训练完后评估在 target domain 上的准确率
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for data, label in target_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)[0]
            prediction = output.argmax(dim=1, keepdim=True)
            accuracy += prediction.eq(label.view_as(prediction)).sum().item()
    accuracy /= len(target_loader.dataset)

    # 打印每个 Epoch 的信息
    print(f"Epoch [{epoch + 1}/{num_epochs}] | "
          f"分类损失: {total_class_loss / len_dataloader:.4f} | "
          f"领域损失: {total_domain_loss / len_dataloader:.4f} | "
          f"准确率: {accuracy:.4f} | "
          f"当前 Alpha: {alpha:.4f}")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), './best_model.pth')
        print(f"最佳模型已保存，准确率: {best_accuracy:.4f}")