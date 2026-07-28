"""
Filter Explanation: 输入一张图，观察某一个滤波器到底学会了检测什么特征。
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = models.resnet34(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 11)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
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

# 预处理并加上 batch 维度: (1, 3, 512, 512)
input_tensor = transform(img).unsqueeze(0).to(device)

# 3. 设置 Hook
activations = {}
def get_activation(name):
    # 这个钩子会在目标层完成计算后，把输出存进字典里
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

"""
ResNet34 所有可以挂钩子 (Hook) 并且易于画图的层级代码
注意：换了层之后，提取特征的代码也要跟着改字典里的名字！
比如用了 model.layer4[0].conv1，名字叫 'layer4_0_conv1'
后面拿特征图的代码就必须写成：
feature_maps = activations['layer4_0_conv1'].cpu().numpy()
"""

# 最外层的前置网络，输出通道数：64
handle = model.conv1.register_forward_hook(get_activation('conv1_output'))
# handle = model.bn1.register_forward_hook(get_activation('bn1_output'))
# handle = model.relu.register_forward_hook(get_activation('relu_output'))
# handle = model.maxpool.register_forward_hook(get_activation('maxpool_output'))

# Layer 1，输出通道数：64
# handle = model.layer1[0].conv1.register_forward_hook(get_activation('layer1_0_conv1'))
# handle = model.layer1[0].conv2.register_forward_hook(get_activation('layer1_0_conv2'))
# handle = model.layer1[1].conv1.register_forward_hook(get_activation('layer1_1_conv1'))
# handle = model.layer1[2].conv2.register_forward_hook(get_activation('layer1_2_conv2'))

# layer 2，输出通道数：128，特征图尺寸缩小
# handle = model.layer2[0].conv1.register_forward_hook(get_activation('layer2_0_conv1'))
# handle = model.layer2[0].conv2.register_forward_hook(get_activation('layer2_0_conv2'))
# handle = model.layer2[0].downsample[0].register_forward_hook(get_activation('layer2_0_downsample'))
# handle = model.layer2[3].conv2.register_forward_hook(get_activation('layer2_3_conv2'))

# Layer 3，输出通道数：256，特征图尺寸再次缩小
# handle = model.layer3[0].conv1.register_forward_hook(get_activation('layer3_0_conv1'))
# handle = model.layer3[2].conv1.register_forward_hook(get_activation('layer3_2_conv1'))
# handle = model.layer3[5].conv2.register_forward_hook(get_activation('layer3_5_conv2'))

# Layer 4，输出通道数：512，特征图尺寸极小
# handle = model.layer4[0].conv1.register_forward_hook(get_activation('layer4_0_conv1'))
# handle = model.layer4[0].downsample[0].register_forward_hook(get_activation('layer4_0_downsample'))
# handle = model.layer4[2].conv2.register_forward_hook(get_activation('layer4_2_conv2'))

# 前向传播，触发 Hook
with torch.no_grad():
    _ = model(input_tensor)

# 拿出被拦截下来的特征图，并转回 CPU 变成 numpy 数组
# ResNet34 的 conv1 输出通道数是 64，所以形状是 (1, 64, 64, 64) -> (Batch, Filters, H, W)
feature_maps = activations['conv1_output'].cpu().numpy()
# 把钩子拆了
handle.remove()

# 画图展示
fig, axs = plt.subplots(1, 3, figsize=(8, 6))

# 原图
axs[0].imshow(img.resize((512, 512)))
axs[0].set_title("Original Image")
axs[0].axis('off')

# 2. 画第 0 个 Filter 的特征图
# 提取索引为 0 的图像，索引为 0 的通道
fmap_0 = feature_maps[0, 0, :, :]
axs[1].imshow(fmap_0)
axs[1].set_title("Filter 0 Feature Map")
axs[1].axis('off')

# 3. 画第 1 个 Filter 的特征图
# 提取索引为 0 的图像，索引为 1 的通道
fmap_1 = feature_maps[0, 1, :, :]
axs[2].imshow(fmap_1)
axs[2].set_title("Filter 1 Feature Map")
axs[2].axis('off')

plt.tight_layout()
plt.show()