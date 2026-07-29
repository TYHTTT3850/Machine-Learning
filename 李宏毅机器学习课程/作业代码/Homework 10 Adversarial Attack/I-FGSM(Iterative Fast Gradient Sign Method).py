"""
======================================================================
I-FGSM (Iterative Fast Gradient Sign Method)
步骤：
1. 前向传播：计算模型对当前受攻击图片的预测 Loss。
2. 反向传播：计算 Loss 对当前图片本身的梯度。
3. 提取符号并走一小步：使用 .sign() 提取梯度的正负号，乘以步长 alpha。
4. 限制累计扰动不超过 epsilon，并为下一次迭代准备。
数学公式：X_{adv}^{N+1} = Clip_{X, epsilon} { X_{adv}^N + alpha * sign(∇ loss) }
======================================================================
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# 设置模型保存路径
model_dir = './pretrained_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.hub.set_dir(model_dir)

# 确保保存图片的输出文件夹存在
output_dir = './I-FGSM'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取本地图片并进行预处理
input_image = Image.open("raw_dog.jpg")

# ImageNet 标准预处理流程
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 预处理图片，增加 Batch 维度，并移动到 GPU
image_tensor = preprocess(input_image).unsqueeze(0).to(device)

# 加载模型
model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
model.eval()

# 真实标签也送到 GPU 上(使用的图片是萨摩耶，在ImageNet中对应类别索引为258)
original_label = torch.tensor([258]).to(device)

# 正常预测(攻击前)
output = model(image_tensor)
criterion = nn.CrossEntropyLoss()

print(f"攻击前模型预测类别: {output.argmax().item()}")

# I-FGSM 攻击生成对抗样本
epsilon = 0.1
num_steps = 10  # 迭代次数
alpha = (epsilon / num_steps) * 1.5  # 每一步的步长

# 复制原图作为攻击起点，并断开原图的梯度追踪
attacked_image_tensor = image_tensor.clone().detach()

for step in range(num_steps):
    # 每次迭代都需要开启梯度追踪
    attacked_image_tensor.requires_grad_()

    # 前向传播
    output = model(attacked_image_tensor)
    loss = criterion(output, original_label)

    # 清空梯度，反向传播
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        # 走一小步
        temp_image_tensor = attacked_image_tensor + alpha * attacked_image_tensor.grad.sign()

        # 计算当前图与原图的累计差异，强行限制在 [-epsilon, epsilon] 之间
        eta = torch.clamp(temp_image_tensor - image_tensor, min=-epsilon, max=epsilon)

        #将合法的差异加上去
        attacked_image_tensor = image_tensor + eta

        # 断开计算图，为下一次循环做准备
        attacked_image_tensor = attacked_image_tensor.detach()


# 测试对抗样本(攻击后)
with torch.no_grad():
    attacked_output = model(attacked_image_tensor)
print(f"攻击后模型预测类别变成了: {attacked_output.argmax().item()}")

# 移动到 CPU 并去掉 Batch 维度
attacked_image_tensor = attacked_image_tensor.detach().cpu().squeeze(0)
image_tensor = image_tensor.detach().cpu().squeeze(0)

# 攻击后和原来的差异
noise_tensor = attacked_image_tensor - image_tensor

# 转换维度：(C, H, W) -> (H, W, C)，并转为 numpy 数组供 matplotlib 使用
noise_img = noise_tensor.permute(1, 2, 0).numpy()

# 将微小的扰动值拉伸到 0.0 ~ 1.0 的范围，这样原本肉眼看不见的噪点就会变成明显的花纹
noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())

# 绘制差异图
plt.figure(figsize=(8, 6))
plt.imshow(noise_img)
plt.title("Difference between original and attacked image (I-FGSM)")
plt.axis('off')  # 隐藏坐标轴
plt.savefig(f"{output_dir}/difference_dog.jpg")

# 反归一化
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
attacked_image_tensor = attacked_image_tensor * std + mean
image_tensor = image_tensor * std + mean

# 确保被攻击的图像还原后的像素值严格在 0 到 1 之间，防止越界导致图片出现奇怪的色块
attacked_image_tensor = torch.clamp(attacked_image_tensor, 0, 1)

# 将 PyTorch 张量转换为普通的 PIL 图像并保存
to_pil = transforms.ToPILImage()
original_image = to_pil(image_tensor)
original_image.save(f"{output_dir}/input_dog.jpg")
attacked_image = to_pil(attacked_image_tensor)
attacked_image.save(f"{output_dir}/attacked_dog.jpg")