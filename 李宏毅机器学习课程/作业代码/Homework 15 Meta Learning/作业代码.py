import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import copy
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

random.seed(42)

# 1. 数据处理与 Task 采样器
def get_omniglot_dict(dataset):
    """将 Omniglot 数据集按类别组织成字典，方便采样"""
    class_dict = defaultdict(list)
    for i in range(len(dataset)):
        img, label = dataset[i]
        class_dict[label].append(img)
    return class_dict

def sample_task(class_dict, n_way, k_shot, q_query):
    """
    采样一个 N-way K-shot 的 Task
    返回: support_x, support_y, query_x, query_y
    """
    # 随机抽取 N 个类别
    classes = random.sample(list(class_dict.keys()), n_way)
    support_x, support_y = [], []
    query_x, query_y = [], []

    for i, cls in enumerate(classes):
        # 从该类别中抽取 K + Q 张图片
        imgs = random.sample(class_dict[cls], k_shot + q_query)
        support_x.extend(imgs[:k_shot])
        query_x.extend(imgs[k_shot:])
        # 标签重置为 0 ~ N-1
        support_y.extend([i] * k_shot)
        query_y.extend([i] * q_query)

    support_x = torch.stack(support_x)#把 support_x 从由 tensor 组成的 python 列表转换为形状为 [N*K, 1, 28, 28] 的 tensor
    query_x = torch.stack(query_x)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_y = torch.tensor(query_y, dtype=torch.long)

    # 打乱 Support 集和 Query 集
    s_indices = torch.randperm(n_way * k_shot)
    q_indices = torch.randperm(n_way * q_query)

    return support_x[s_indices], support_y[s_indices], query_x[q_indices], query_y[q_indices]


# 2. 模型定义
class SimpleCNN(nn.Module):
    """用于 Omniglot 的标准 4 层 CNN"""
    def __init__(self, n_way):
        super(SimpleCNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c, track_running_stats=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(1, 64),  # 28x28 -> 14x14
            conv_block(64, 64),  # 14x14 -> 7x7
            conv_block(64, 64),  # 7x7 -> 3x3
            conv_block(64, 64)  # 3x3 -> 1x1
        )
        self.classifier = nn.Linear(64, n_way)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 3. 超参数设置与数据准备
N_WAY = 5
K_SHOT = 1  # 5-way 1-shot
Q_QUERY = 5  # 每个类别 query 样本数
TASKS_PER_META_BATCH = 16  # 一个 meta batch 有几个任务
INNER_STEPS = 1  # 内循环(一个训练任务)更新几次参数
INNER_LR = 0.4  # 内循环(一个训练任务)学习率
META_LR = 0.001  # 外循环(元)学习率
EPOCHS = 50  # 元训练轮数
META_BATCHES_PER_EPOCH = 100  # 每轮包含多少个 meta-batch

# 预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

print("加载 Omniglot 数据集...")
train_dataset = torchvision.datasets.Omniglot(root='./data', background=True, download=True, transform=transform)
test_dataset = torchvision.datasets.Omniglot(root='./data', background=False, download=True, transform=transform)

train_dict = get_omniglot_dict(train_dataset)
test_dict = get_omniglot_dict(test_dataset)

# 初始化 Meta 模型和优化器
meta_model = SimpleCNN(N_WAY).to(device)
meta_optimizer = optim.Adam(meta_model.parameters(), lr=META_LR)
criterion = nn.CrossEntropyLoss()


# 4. 元训练主循环
for epoch in range(EPOCHS):
    meta_model.train()

    # 重置一个 meta batch 的损失和准确率
    meta_batch_loss = 0.0
    meta_batch_acc = 0.0
    for step in range(META_BATCHES_PER_EPOCH):
        meta_optimizer.zero_grad()

        batch_loss = 0.0
        batch_acc = 0.0

        # 外循环 (Meta-Batch)
        for _ in range(TASKS_PER_META_BATCH):
            s_x, s_y, q_x, q_y = sample_task(train_dict, N_WAY, K_SHOT, Q_QUERY)
            s_x, s_y = s_x.to(device), s_y.to(device)
            q_x, q_y = q_x.to(device), q_y.to(device)

            # 复制模型，准备内循环
            fast_model = copy.deepcopy(meta_model)
            fast_optimizer = optim.SGD(fast_model.parameters(), lr=INNER_LR)

            # 内循环 (Inner Loop)
            for _ in range(INNER_STEPS):
                s_logits = fast_model(s_x)
                inner_loss = criterion(s_logits, s_y)

                fast_optimizer.zero_grad()
                inner_loss.backward()
                fast_optimizer.step()

            # 在 Query 集上计算损失
            q_logits = fast_model(q_x)
            q_loss = criterion(q_logits, q_y)

            # FOMAML 一阶导数计算
            q_loss.backward()

            # 记录本 task 的表现
            batch_loss += q_loss.item()
            preds = q_logits.argmax(dim=1)
            batch_acc += (preds == q_y).float().mean().item()

            # 梯度累加到 meta_model(因为数据不是从 meta_model 开始的，而是从深拷贝得到的 fast_model 开始的，反向传播到 fast_model 后就停止了，meta_model 并没有梯度)
            for meta_param, fast_param in zip(meta_model.parameters(), fast_model.parameters()):
                if fast_param.grad is not None:
                    if meta_param.grad is None:
                        meta_param.grad = fast_param.grad.clone() / TASKS_PER_META_BATCH # 训练任务的平均梯度
                    else:
                        meta_param.grad += fast_param.grad / TASKS_PER_META_BATCH # 训练任务的平均梯度

        # 针对当前 meta-batch 更新元模型参数
        meta_optimizer.step()
        meta_batch_loss += batch_loss / TASKS_PER_META_BATCH # 一个 meta batch 中所有训练任务的平均损失
        meta_batch_acc += batch_acc / TASKS_PER_META_BATCH # 一个 meta batch 中所有训练任务的平均准确率

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {meta_batch_loss / META_BATCHES_PER_EPOCH:.4f} | Train Acc: {meta_batch_acc / META_BATCHES_PER_EPOCH:.4f}")

    # 5. 模型测试阶段 (每 5 个 Epoch 测试一次)
    if (epoch + 1) % 5 == 0:
        meta_model.eval()
        test_acc = 0.0
        test_tasks = 200  # 评估采样 200 个任务

        for _ in range(test_tasks):
            s_x, s_y, q_x, q_y = sample_task(test_dict, N_WAY, K_SHOT, Q_QUERY)
            s_x, s_y = s_x.to(device), s_y.to(device)
            q_x, q_y = q_x.to(device), q_y.to(device)

            test_fast_model = copy.deepcopy(meta_model)
            test_fast_optimizer = optim.SGD(test_fast_model.parameters(), lr=INNER_LR)
            test_fast_model.train()
            
            # 测试时参数可以进行多次更新
            for _ in range(3):
                s_logits = test_fast_model(s_x)
                inner_loss = criterion(s_logits, s_y)
                test_fast_optimizer.zero_grad()
                inner_loss.backward()
                test_fast_optimizer.step()

            test_fast_model.eval()
            with torch.no_grad():
                q_logits = test_fast_model(q_x)
                preds = q_logits.argmax(dim=1)
                test_acc += (preds == q_y).float().mean().item() # 一个测试任务的准确率

        print(f">>> Test Acc (5-way 1-shot): {test_acc / test_tasks:.4f}")
        print("-" * 50)