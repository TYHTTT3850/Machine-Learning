import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化一个 ResNet34 (这里 weights=None 即可，因为要加载自己的权重)
model = models.resnet34(weights=None)

# 修改最后一层全连接层，适配 Food-11 的 11 个类别
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 11)

# 导入保存的参数
# map_location 可以保证即使是在 GPU 训练的，也能顺利在 CPU 上加载测试
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint)

# 放到对应设备并开启 eval 模式 (关闭 Dropout 和 BatchNorm 的动态变化)
model = model.to(device)
model.eval()

# 预处理流程 (与验证集的预处理保持一致)
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小为 512x512
    transforms.ToTensor(),
    transforms.Normalize(  # 归一化
        mean=[0.485, 0.456, 0.406],  # RGB 3通道的均值
        std=[0.229, 0.224, 0.225]  # RGB 3通道的标准差
    )
])

image_path = 'food11/validation/10/10_0.jpg'  # 替换成你要解释的图片路径
img = Image.open(image_path).convert('RGB')

# 加上 batch size 维度 (Channel, Height, Weight) -> (1, Channel, Height, Weight) 并且放到 GPU/CPU 上
input_tensor = transform(img).unsqueeze(0).to(device)

# 第三部分：计算 Saliency Map
# 开启对输入图片的梯度追踪
input_tensor.requires_grad_()

# 清空梯度
model.zero_grad()

# 前向传播得到 11 个类别的分数
logits = model(input_tensor)

# 找到模型预测出来的最高分的类别索引
target_class_index = logits.argmax(dim=1).item()

# 提取出那个最高分
score = logits[0, target_class_index]

# 反向传播，计算分数对 input_tensor 每一个像素的偏导数
score.backward()


# 提取梯度并画图
# 取出图片上的梯度，并转回 CPU 去掉 batch 维度
saliency = input_tensor.grad.data.squeeze().cpu()

# 取绝对值
saliency = saliency.abs()
# 在 3 个颜色通道(dim=0)中取最大值
saliency, _ = torch.max(saliency, dim=0)
# 转换为 numpy 数组
saliency = saliency.numpy()
# 4. 归一化到 0~1，方便画图
saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

# 开始画图对比
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# 左边画原图
ax[0].imshow(img.resize((512, 512)))
ax[0].set_title("Original Image")
ax[0].axis('off')

# 右边画显著图
ax[1].imshow(saliency, cmap='hot')
ax[1].set_title(f"Saliency Map (Class {target_class_index})")
ax[1].axis('off')

plt.show()