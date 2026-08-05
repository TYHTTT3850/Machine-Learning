import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# 超参数设置
LR = 3e-4  # 学习率
GAMMA = 0.99  # 奖励折扣因子
EPS_CLIP = 0.2  # PPO Clip 的截断范围
K_EPOCHS = 10  # 每次更新时，复用数据的训练轮数
UPDATE_TIMESTEPS = 4000  # 每收集 4000 步数据更新一次网络
MAX_EPISODES = 3000  # 最大训练回合数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 经验池
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


# Actor-Critic 神经网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor 网络 (输出动作概率)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic 网络 (评估状态价值)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        # 将 numpy 数组转为 tensor
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)

        # 获取动作概率并采样
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        # 记录状态、动作和它的对数概率
        memory.states.append(state.squeeze(0))
        memory.actions.append(action.squeeze(0))
        memory.logprobs.append(dist.log_prob(action).squeeze(0))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy



# PPO 算法
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        # PPO 需要一个旧策略来计算 Ratio
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        #计算带折扣的累计奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        #对 Reward 进行标准化
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 将经验池中的列表转换为 Tensor
        old_states = torch.stack(memory.states).detach().to(device)
        old_actions = torch.stack(memory.actions).detach().to(device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(device)

        # 在循环外，使用旧网络计算一次当前状态的 Value
        _, old_state_values, _ = self.policy_old.evaluate(old_states, old_actions)

        # 计算优势函数 Advantage = 实际回报 - 预期回报
        advantages = rewards - old_state_values.detach()

        #对 Advantage 进行标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 3. K_EPOCHS 轮优化
        for _ in range(K_EPOCHS):
            # 获取新策略的评估
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算新旧策略的比率 (Ratio)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算 Surrogate Loss (带有 Clip 截断)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            # Actor 损失 + Critic 损失 + 鼓励探索的熵奖励
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 反向传播更新网络
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())



# 训练主流程
env = gym.make('LunarLander-v3')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

memory = Memory()
ppo = PPO(state_dim, action_dim)

time_step = 0
running_reward = 0
best_avg_reward = -float('inf')  # 用于跟踪最佳平均奖励

print(f"开始训练，使用设备:{device}")
for i_episode in range(1, MAX_EPISODES + 1):
    state, _ = env.reset()
    ep_reward = 0

    while True:
        time_step += 1

        # 与环境交互
        action = ppo.policy_old.act(state, memory)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        ep_reward += reward

        # 达到收集步数，进行网络更新
        if time_step % UPDATE_TIMESTEPS == 0:
            ppo.update(memory)
            memory.clear()
            time_step = 0

        if done:
            break

    running_reward += ep_reward

    # 打印训练日志
    if i_episode % 20 == 0:
        avg_reward = running_reward / 20
        print(f"Episode {i_episode}，平均奖励(最近20局): {avg_reward:.2f}")
        running_reward = 0

        # 保存最佳模型
        if avg_reward > best_avg_reward:
            print(f"新的最佳平均奖励: {avg_reward:.2f}，保存模型权重")
            torch.save(ppo.policy.state_dict(), "ppo_lunar_lander_solved.pth")
            best_avg_reward = avg_reward

env.close()