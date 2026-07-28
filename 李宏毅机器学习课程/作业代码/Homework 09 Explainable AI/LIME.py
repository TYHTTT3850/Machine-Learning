"""
LIME: 把图片切成一堆小碎块(Superpixels)，然后随机把某些块涂黑，送进模型看分数变化。哪个块被涂黑后模型预测分数暴跌，就说明哪个块最重要。
"""
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 导入 LIME 和画图所需的库
from lime import lime_image
from skimage.segmentation import mark_boundaries
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和预处理
model = models.resnet34(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 11)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图像大小为 512x512
    transforms.ToTensor(),
    transforms.Normalize(  # 归一化
        mean=[0.485, 0.456, 0.406],  # RGB 3通道的均值
        std=[0.229, 0.224, 0.225]  # RGB 3通道的标准差
    )
])

# 预测函数
# LIME 框架内部不认识 PyTorch 的 Tensor，只接受 Numpy 的 ndarray。
def batch_predict(images):
    """
    输入: images 是一个形如 (N, H, W, C) 的 numpy 数组 (像素值 0-255)
    输出: 一个形如 (N, 11) 的 numpy 数组，包含所有类别的预测概率
    """
    model.eval()
    batch_tensors = []

    # 遍历 LIME 传进来的每一张扰动图
    for img_array in images:
        # 转回 PIL 图片，然后套用我们之前定义好的 transform
        img_pil = Image.fromarray(img_array.astype('uint8'))
        img_tensor = transform(img_pil)
        batch_tensors.append(img_tensor)

    # 拼装成 Batch 送入 GPU
    batch_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        # 注意！LIME 在底层做线性拟合时需要的是概率分布，不是原始的 Logits
        # 所以必须要加 Softmax 将分数映射到 0~1 之间
        probs = F.softmax(logits, dim=1)

    return probs.cpu().numpy()

# 读取原图并调整尺寸
image_path = 'food11/validation/10/10_0.jpg'
img = Image.open(image_path).convert('RGB')
# LIME 处理时需要一张标准的 Numpy 图片作为基底
img_numpy = np.array(img.resize((512, 512)))

# 4. 初始化 LIME 并生成解释
explainer = lime_image.LimeImageExplainer()

print("LIME 正在生成扰动图片并计算")
# LIME 会把图片切成超像素块，然后随机涂黑组合，调用 1000 次 batch_predict
explanation = explainer.explain_instance(
    image=img_numpy,# 传入要解释的图片
    classifier_fn=batch_predict,  # 传入预测函数
    top_labels=1,  # 只关心预测概率最高的那个类别的解释
    hide_color=0,  # 把超像素块遮挡成纯黑色 (0)
    num_samples=1000  # 采样数量1000，想更快可以改小，想更准可以改大
)


# 提取并画出最终解释结果
# 取出模型最确信的那个类别
target_class = explanation.top_labels[0]

# 从 LIME 结果中拿回被标记的图像和高亮遮罩 (Mask)
# positive_only=True: 只看对最终决定起“正面推动作用”的图像块
# num_features=5: 只展示贡献最大的前 5 个像素块
# hide_rest=False: 是否把不重要的背景全涂黑(选 False 可以在原图上画框，更直观)
temp, mask = explanation.get_image_and_mask(
    label=target_class,
    positive_only=True,
    num_features=5,
    hide_rest=False
)

# mask 是一个二维数组，重要区域的值是 1，不重要的区域是 0。把值为 0 的地方透明化，只保留值为 1 的地方
overlay_mask = np.ma.masked_where(mask == 0, mask)

# 画图
fig, ax = plt.subplots(1, 2, figsize=(8, 6))

# 原图对比
ax[0].imshow(img_numpy)
ax[0].set_title("Original Image")
ax[0].axis('off')

# LIME 结果
ax[1].imshow(img_numpy)
ax[1].imshow(overlay_mask, cmap='cool', alpha=1, interpolation='none')
ax[1].set_title(f"LIME Explanations (Class {target_class})")
ax[1].axis('off')

plt.tight_layout()
plt.show()