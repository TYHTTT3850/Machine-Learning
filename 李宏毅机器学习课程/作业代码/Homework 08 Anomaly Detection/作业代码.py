import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==========================================
# 1. 自定义数据集 (处理 NumPy 数组并转换维度)
# ==========================================
class ImageAnomalyDataset(Dataset):
    def __init__(self, data_array):
        """
        data_array: shape (N, 64, 64, 3) 的 numpy 数组
        """
        self.data = data_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]

        # 归一化到 [0, 1]
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        # PyTorch 需要 (Channel, Height, Width) 格式
        img = np.transpose(img, (2, 0, 1))

        # 转换为 Tensor
        return torch.tensor(img, dtype=torch.float32)


# ==========================================
# 2. 定义自编码器
# ==========================================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        """
        latent_dim: 中间特征向量的维度，维度越低压缩越厉害
        """
        super(Autoencoder, self).__init__()

        # 1. 卷积编码部分
        # 输入: (3, 64, 64) -> 输出: (64, 8, 8)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 8, 8)
            nn.ReLU(True)
        )

        # 2. 向量化编码部分
        # 展平后大小为 64 * 8 * 8 = 4096
        self.encoder_fc = nn.Linear(64 * 8 * 8, latent_dim)

        # 3. 向量化解码部分
        # 将低维向量还原回 4096 维
        self.decoder_fc = nn.Linear(latent_dim, 64 * 8 * 8)

        # 4. 卷积解码部分
        # 输入: (64, 8, 8) -> 输出: (3, 64, 64)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (16, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (3, 64, 64)
            nn.Sigmoid() #归一化到[0,1]
        )

    def forward(self, x):
        # 编码
        x = self.encoder_conv(x)  # 提取特征图: (Batch, 64, 8, 8)
        x = torch.flatten(x, start_dim=1)  # 展平: (Batch, 4096)
        latent_vector = self.encoder_fc(x)  # 压缩为低维向量: (Batch, latent_dim)

        # 解码
        x = self.decoder_fc(latent_vector)  # 还原维度: (Batch, 4096)
        x = x.view(-1, 64, 8, 8)  # 变回特征图形状: (Batch, 64, 8, 8)
        reconstructed = self.decoder_conv(x)  # 上采样还原图像: (Batch, 3, 64, 64)

        return reconstructed

# ==========================================
# 5. 训练循环及推理
# ==========================================
if __name__ == "__main__":
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    raw_training_data = np.load("trainingset.npy")
    print(f"Loaded data shape: {raw_training_data.shape}")

    # 准备 DataLoader
    train_dataset = ImageAnomalyDataset(raw_training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 初始化模型并移至设备
    model = Autoencoder().to(device)

    # 训练模型
    epochs = 20
    print(f"开始训练，使用设备{device}")
    criterion = nn.MSELoss()  # 使用均方误差作为重构损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_dataloader:
            data = data.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, data)  # 计算当前批次中，所有的像素点的 MSE 误差的平均值(注意此处会除以 batch size)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*data.size(0)# 将 batch size 乘回来得到当前批次中每张图的 MSE 误差，然后求和

        epoch_loss = running_loss / len(train_dataset)# 总体的 MSE 误差
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss}")

    # 计算所有训练数据的重构误差
    model.eval()  # 设置模型为评估模式
    reconstruction_errors = []

    # 不计算梯度
    with torch.no_grad():
        for data in train_dataloader:
            data = data.to(device)
            outputs = model(data)

            # 计算每张图片的 MSE 误差
            # data 尺寸: (Batch, 3, 64, 64)
            # batch_errors 尺寸: (Batch,)
            batch_errors = torch.mean((data - outputs) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(batch_errors.cpu().numpy())

    # 设定异常阈值，取训练集误差的 95% 分位数
    # 误差大于阈值的图片就是"异常"的
    train_errors = np.array(reconstruction_errors)
    threshold = np.percentile(train_errors, 95)
    print(f"阈值为：{threshold}")

    # 筛选测试集中的异常
    raw_testing_data = np.load("testingset.npy")
    test_dataset = ImageAnomalyDataset(raw_testing_data)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    reconstruction_errors = []
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            outputs = model(data)
            batch_errors = torch.mean((data - outputs) ** 2, dim=[1, 2, 3])
            reconstruction_errors.extend(batch_errors.cpu().numpy())

    # 根据阈值，筛选测试集中哪些图片是异常的
    test_errors = np.array(reconstruction_errors)
    anomalies = test_errors > threshold
    # 输出为 csv
    np.savetxt("anomalies.csv", anomalies, delimiter=",", fmt="%d", header="is_anomaly", comments="")
