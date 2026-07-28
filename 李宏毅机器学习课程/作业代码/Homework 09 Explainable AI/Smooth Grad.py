"""
Smooth Grad：在某一图片上面加上各种不同的噪声，得到各种不同的图片后，对每一张图片计算 Saliency Map，平均起来得到 Smooth Grad 的结果，这样得到的结果往往能够更加集中在被侦测的物体上。
"""
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

image_path = 'food11/validation/10/10_0.jpg'
img = Image.open(image_path).convert('RGB')

# 加上 batch size 维度 (Channel, Height, Weight) -> (1, Channel, Height, Weight) 并且放到 GPU/CPU 上
input_tensor = transform(img).unsqueeze(0).to(device)

model.eval()
input_tensor = input_tensor.to(device)

# 用干净的原图做一次前向传播，拿到预测分数最高的类别
with torch.no_grad():
    clean_logits = model(input_tensor)
    target_class_index = clean_logits.argmax(dim=1).item()

# SmoothGrad
num_samples = 50  # 采样次数：要加多少次噪声
stdev_spread = 0.25  # 噪声强度：标准差的比例（通常 0.1 到 0.2）

# 动态计算高斯噪声的标准差 std
# 公式：std = stdev_spread * (图片最大值 - 图片最小值)
std = stdev_spread * (input_tensor.max() - input_tensor.min()).item()

# 准备一个全零的张量，用来累加这 50 次算出来的梯度
# 形状和 input_tensor 一样
total_gradients = torch.zeros_like(input_tensor)

# 开始循环：每次加不同的噪声，算梯度，累加
for i in range(num_samples):
    # 1. 生成与输入形状相同的高斯噪声
    noise = torch.normal(mean=0.0, std=std, size=input_tensor.shape).to(device)

    # 2. 将噪声叠加到原图上 (注意：每次都是在原始 input_tensor 上加新的噪声)
    noisy_img = input_tensor + noise

    # 3. 开启对这张噪声图的梯度追踪
    noisy_img.requires_grad_()

    # 4. 清空模型梯度
    model.zero_grad()

    # 5. 前向传播
    logits = model(noisy_img)

    # 6. 提取目标类别的分数 (必须固定是刚才确定的 target_class_idx)
    score = logits[0, target_class_index]

    # 7. 反向传播
    score.backward()

    # 8. 累加梯度 (注意要用 .data，避免将计算图也保存下来导致内存泄漏)
    total_gradients += noisy_img.grad.data

# 求平均梯度
smooth_saliency = total_gradients / num_samples


# 去掉 batch 维度，保留 3 个颜色通道 (C, H, W)
smooth_saliency_rgb = smooth_saliency.squeeze().cpu()

# 2. 取绝对值
smooth_saliency_rgb = smooth_saliency_rgb.abs()

# 3. 转换维度顺序，适配 Matplotlib 的格式需求 (C, H, W) -> (H, W, C)
smooth_saliency_rgb = smooth_saliency_rgb.permute(1, 2, 0).detach().numpy()

# 4. 把这个 3 通道的三维数组整体归一化到 0~1 之间
smooth_saliency_rgb_normalized = (smooth_saliency_rgb - smooth_saliency_rgb.min()) / (smooth_saliency_rgb.max() - smooth_saliency_rgb.min() + 1e-8)

# 开始画图
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# 左侧：原图
ax[0].imshow(img.resize((512, 512)))
ax[0].set_title("Original Image")
ax[0].axis('off')

# 右侧： SmoothGrad 效果
ax[1].imshow(smooth_saliency_rgb_normalized)
ax[1].set_title(f"SmoothGrad Color Effect (Samples: {num_samples})")
ax[1].axis('off')

plt.show()