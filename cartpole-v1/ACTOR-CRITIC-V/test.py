import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Pi(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Pi, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        action_probs = torch.softmax(self.linear2(x), dim=-1)
        return action_probs

class QNet(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(QNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        state_value = self.linear2(x)
        return state_value
    
policy = Pi(4, 2, 256)
value_net = QNet(4, 256)
policy.load_state_dict(torch.load('policy.pth'))
value_net.load_state_dict(torch.load('value_net.pth'))

env = gym.make('CartPole-v1', render_mode='human')

state = env.reset()
done = False
total_reward = 0
for _ in range(10000):
    if type(state) is tuple:
        state, _  = state
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy(state)
    values = value_net(state)
    m = Categorical(probs)
    action = m.sample()
    next_state, reward, done, _, _ = env.step(action.item())
    total_reward += reward
    env.render()
    state = next_state
    if done:
        break
env.close()