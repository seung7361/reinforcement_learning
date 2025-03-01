import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(4, 256),  # CartPole observation space is 4
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, 2)  # CartPole action space is 2
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.common(x)
        action_probs = self.actor(x)
        state_values = self.critic(x)
        return torch.softmax(action_probs, dim=-1), state_values

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

env = gym.make('CartPole-v1')
model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

total_rewards = []

num_episodes = 2000
for episode in range(num_episodes):
    state = env.reset()
    log_probs = []
    values = []
    rewards = []
    masks = []
    done = False
    while not done:
        if type(state) is tuple:
            state, _ = state
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = model(state)
        m = Categorical(probs)
        action = m.sample()

        next_state, reward, done, _, _ = env.step(action.item())

        log_prob = m.log_prob(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        masks.append(torch.tensor([1-done], dtype=torch.float))

        state = next_state

    next_state = torch.FloatTensor(next_state).unsqueeze(0)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    optimizer.step()

    print(f"Episode: {episode + 1}, Total reward: {sum(rewards)}")
    total_rewards.append(sum(rewards))

env.close()

# Plot the total rewards
import matplotlib.pyplot as plt

plt.plot(total_rewards, label="Total rewards", color="blue")
ewa = [total_rewards[0]]
for r in total_rewards[1:]:
    ewa.append(0.05 * r + 0.95 * ewa[-1])
plt.plot(ewa, label="Exponential weighted average", color="red")
plt.xlabel("Episode")
plt.ylabel("Total reward")
plt.legend()
plt.savefig("total_reward.png")