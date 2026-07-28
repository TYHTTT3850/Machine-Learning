"""
Integrated Gradient: 设定一个基准图像，通常是一张全黑的图片(像素值全为 0)。让基准图像一步步过渡到真实的输入图像。在这个过渡的路径上，切分出 m 个步长(比如 50 步)。计算每一步的图像的 Saliency map，并把它们求和平均起来。最后，用真实图像减去基准图像，乘以这个平均梯度。
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型与预处理
model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 11)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path = 'food11/validation/10/10_0.jpg'
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device) #加入 batch size 维度

# 获取目标类别
with torch.no_grad():
    clean_logits = model(input_tensor)
    target_class_index = clean_logits.argmax(dim=1).item()


# Integrated Gradients
steps = 50  # 过渡的步数

# 定义基准图像，通常使用与输入形状相同的全零张量 (全黑图像)
baseline = torch.zeros_like(input_tensor).to(device)

# 用来累加这 50 步的梯度
total_gradients = torch.zeros_like(input_tensor).to(device)

# 计算每一步的 saliency map
for i in range(1, steps + 1):
    # 计算当前的插值系数 alpha (从 1/50 逐渐增加到 1.0)
    alpha = i / steps

    # 按照公式生成介于 baseline 和 input_tensor 之间的插值图像
    interpolated_img = baseline + alpha * (input_tensor - baseline)

    # 开启梯度追踪
    interpolated_img.requires_grad_()

    # 清空模型梯度
    model.zero_grad()

    # 前向传播并求导
    logits = model(interpolated_img)
    score = logits[0, target_class_index]
    score.backward()

    # 累加这步的梯度
    total_gradients += interpolated_img.grad.data

# 计算平均梯度
avg_gradients = total_gradients / steps

# 2. 乘以输入与基准的差值，得到最终的 Integrated Gradients
ig_saliency = (input_tensor - baseline) * avg_gradients


# 画图
# 去掉 batch 维度，转回 CPU
ig_saliency = ig_saliency.squeeze().cpu()

# 取绝对值
ig_saliency = ig_saliency.abs()

# 在通道维度取最大值，将 3 个通道压扁成 1 个通道的热力图
ig_saliency, _ = torch.max(ig_saliency, dim=0)
ig_saliency = ig_saliency.numpy()

# 全局归一化到 0~1
ig_saliency = (ig_saliency - ig_saliency.min()) / (ig_saliency.max() - ig_saliency.min() + 1e-8)



# 画图对比
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# 左侧画原图
ax[0].imshow(img.resize((512, 512)))
ax[0].set_title(f"Original Image,class {target_class_index}")
ax[0].axis('off')

# 右侧画积分梯度热力图
ax[1].imshow(ig_saliency, cmap='hot')
ax[1].set_title(f"Integrated Gradients (Steps: {steps})")
ax[1].axis('off')

plt.tight_layout()
plt.show()