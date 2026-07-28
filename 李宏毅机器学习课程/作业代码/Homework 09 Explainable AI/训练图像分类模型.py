import os
os.environ["TORCH_HOME"] = "./pretrained_models" #改变预训练模型下载路径

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models

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

# 引入预训练的 resnet 34
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
num_classes = 11
num_features = model.fc.in_features # 获取全连接层的输入特征数
model.fc = torch.nn.Linear(num_features, num_classes) # 替换为新全连接层
model.to(device)

# 训练循环
epochs = 50
best_validation_accuracy = 0
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    model.train()
    epoch_total_loss = 0
    for images, labels in train_loader:
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()

    # 每一轮训练完后评估
    model.eval()
    correct = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            images,labels = images.to(device),labels.to(device)
            output = model(images)
            predict = output.argmax(dim=1, keepdim=True)
            correct += predict.eq(labels.view_as(predict)).sum().item()
        accuracy = correct / len(validation_set)
        print(f"Epoch: {epoch+1}/{epochs}, epoch_total_loss: {epoch_total_loss}, accuracy：{accuracy}")

        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            torch.save(model.state_dict(), './best_model.pth')
            print(f"Best model saved at epoch {epoch + 1}, accuracy: {accuracy}",end="\n\n")