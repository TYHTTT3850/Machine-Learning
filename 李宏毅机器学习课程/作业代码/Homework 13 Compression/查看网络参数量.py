import torch
import torch.nn as nn
import torchvision.models as models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    # 统计所有参数(包含冻结的参数)
    total_params = sum(p.numel() for p in model.parameters())

    # 仅统计可训练的参数（requires_grad=True）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

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

# 1. 加载教师模型
teacher_model = models.resnet18(weights=None)
num_classes = 11
num_features = teacher_model.fc.in_features # 获取全连接层的输入特征数
teacher_model.fc = torch.nn.Linear(num_features, num_classes) # 替换为新全连接层
# 导入保存的参数
checkpoint = torch.load('best_teacher_model.pth', map_location=device)
teacher_model.load_state_dict(checkpoint)

# 2. 加载学生模型
student_model = StudentNet(num_classes=11)
# 导入保存的参数
checkpoint = torch.load('best_student_model.pth', map_location=device)
student_model.load_state_dict(checkpoint)

# 3. 统计参数量
t_total, t_trainable = count_parameters(teacher_model)
s_total, s_trainable = count_parameters(student_model)

print(f"Teacher 模型总参数量: {t_total:,} (可训练: {t_trainable:,})")
print(f"Student 模型总参数量: {s_total:,} (可训练: {s_trainable:,})")

# 计算压缩率
compression_ratio = t_total / s_total
print(f"压缩倍率: {compression_ratio:.2f}x")