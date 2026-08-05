import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# 复制原来的 ActorCritic 网络结构(必须和训练时一模一样，否则无法加载权重)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

# 观看测试
#设置 render_mode="human" 来弹出游戏窗口
env = gym.make('LunarLander-v3', render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 实例化网络
policy = ActorCritic(state_dim, action_dim)

# 加载训练好的模型权重
print("正在加载模型权重...")
try:
    policy.load_state_dict(torch.load("ppo_lunar_lander_solved.pth", weights_only=True))
    print("加载成功！开始演示...")
except FileNotFoundError:
    print("未找到模型文件，请确保先运行了训练代码并成功保存了 .pth 文件。")
    exit()

# 测试 5 局
for ep in range(1, 6):
    state, _ = env.reset()
    ep_reward = 0
    done = False

    while not done:
        # 将状态转换为 Tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # 在测试阶段，不需要探索，直接选择概率最大的动作
        with torch.no_grad():
            action_probs = policy.actor(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()

        # 执行动作
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward

    print(f"第 {ep} 局结束，总得分: {ep_reward:.2f}")

env.close()