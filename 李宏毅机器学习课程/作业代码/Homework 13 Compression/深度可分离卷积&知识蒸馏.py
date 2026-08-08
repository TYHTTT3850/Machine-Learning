import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练时的图像预处理
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),# 调整图像大小为 512x512
    transforms.RandomVerticalFlip(),# 随机垂直翻转
    transforms.RandomHorizontalFlip(p=0.5),# 随机水平翻转
    transforms.ColorJitter(brightness=0.5),# 随机改变图像的亮度。0.5 意味着亮度会在原始亮度的 0.5 倍（变暗 50%）到 1.5 倍（变亮 50%）之间随机浮动
    transforms.RandomRotation(5),# 在 -5 度到 +5 度之间随机旋转图片
    transforms.ToTensor(),# 转换为张量
    transforms.Normalize(#归一化
        mean=[0.485, 0.456, 0.406],  # RGB 3通道的均值
        std=[0.229, 0.224, 0.225]  # RGB 3通道的标准差
    )
])

# 验证时的图像预处理
validation_transform = transforms.Compose([
    transforms.Resize((512, 512)),# 调整图像大小为 512x512
    transforms.ToTensor(),
    transforms.Normalize(#归一化
        mean=[0.485, 0.456, 0.406],  # RGB 3通道的均值
        std=[0.229, 0.224, 0.225]  # RGB 3通道的标准差
    )
])

# ImageFolder 每次遍取数据时，默认都会返回一个元组：(数据, 标签)，返回的标签是一个整数，一一对应文件夹下的类别
train_set = datasets.ImageFolder(root="food11/training", transform=train_transform)
validation_set = datasets.ImageFolder(root="food11/validation", transform=validation_transform)

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. Depthwise 卷积
        # groups=in_channels 是深度可分离卷积的核心
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3,
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU6(inplace=True)  # 使用 ReLU6 更有利于低精度计算和模型压缩

        # 2. Pointwise 卷积：利用 1x1 卷积融合通道特征
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# 利用搭建好的 Block 组建 Student 模型
class StudentNet(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        # 初始层通常还是用标准卷积来提取丰富的低级特征
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # 中间特征提取部分全部替换为深度可分离卷积
        self.features = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2)
        )

        # 全局平均池化 + 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def loss_fn_kd(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    """
    计算知识蒸馏的组合 Loss，知识蒸馏的 Loss 需要同时计算 Student 预测与真实标签(Hard Label)的交叉熵，以及与Teacher 软标签(Soft Label)的 KL 散度。
    :param student_logits: Student 网络的原始输出 (未经 Softmax)
    :param teacher_logits: Teacher 网络的原始输出 (未经 Softmax)
    :param labels: 数据集的真实 hard labels
    :param alpha: Hard label loss 的权重 (通常 0.1~0.5)
    :param temperature: 蒸馏温度 T (通常 3.0~20.0)
    """
    # 1. Hard Loss: 标准的交叉熵 Loss
    hard_loss = F.cross_entropy(student_logits, labels)

    # 2. Soft Loss: KL 散度
    # Teacher 的输出作为目标概率分布，除以 T 并做 Softmax。注意一定要加 .detach() 截断梯度
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1).detach()
    # Student 的输出作为预测概率分布，除以 T 并做 LogSoftmax
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # 计算 KL Divergence (PyTorch要求输入是 log 概率，目标是常规概率)
    # 使用 reduction='batchmean' 保证 batch_size 变化时梯度的稳定性
    soft_loss = F.kl_div(log_soft_student, soft_teacher, reduction='batchmean')

    # 3. 总 Loss 组合
    # 注意 KL 散度项需要乘以 T 的平方，以补偿除以 T 带来的梯度缩放
    loss = alpha * hard_loss + (1.0 - alpha) * (temperature ** 2) * soft_loss

    return loss


# 1. 加载教师模型
teacher_model = models.resnet18(weights=None)
num_classes = 11
num_features = teacher_model.fc.in_features # 获取全连接层的输入特征数
teacher_model.fc = torch.nn.Linear(num_features, num_classes) # 替换为新全连接层
# 导入保存的参数
checkpoint = torch.load('best_teacher_model.pth', map_location=device)
teacher_model.load_state_dict(checkpoint)
teacher_model.to(device)
teacher_model.eval()

student_model = StudentNet(num_classes=11).to(device)
student_model.train()
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)

# 2. 训练循环
epochs = 50
best_validation_accuracy = 0
for epoch in range(epochs):
    epoch_total_loss = 0
    for images, labels in train_loader:  # 使用训练数据加载器
        images, labels = images.to(device), labels.to(device)

        # 获取 Teacher 的知识
        with torch.no_grad():
            teacher_logits = teacher_model(images)

        # 获取 Student 的预测
        student_logits = student_model(images)

        # 计算组合蒸馏 Loss
        # 可以通过实验微调 alpha 和 temperature
        loss = loss_fn_kd(
            student_logits,
            teacher_logits,
            labels,
            alpha=0.3,  # 意味着 30% 靠自己，70% 靠老师
            temperature=5.0  # 软化概率分布
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_total_loss += loss.item()

    # 每一轮训练完后评估
    student_model.eval()
    correct = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            output = student_model(images)
            predict = output.argmax(dim=1, keepdim=True)
            correct += predict.eq(labels.view_as(predict)).sum().item()
        accuracy = correct / len(validation_set)
        print(f"Epoch: {epoch + 1}/{epochs}, epoch_total_loss: {epoch_total_loss}, accuracy：{accuracy}")

        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            torch.save(student_model.state_dict(), './best_student_model.pth')
            print(f"Best model saved at epoch {epoch + 1}, accuracy: {accuracy}", end="\n\n")