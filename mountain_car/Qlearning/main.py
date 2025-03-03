import gymnasium as gym
import numpy as np
import torch
import gymnasium as gym

import random
from collections import deque


### hyperparameters

EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 10
REPLAY_BUFFER_SIZE = 10000


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.model(x)


def epsilon_greedy(state, epsilon, model, out_dim, device):
    if random.random() < epsilon:
        return random.randrange(out_dim)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device) # (1, in_dim)
            q_values = model(state)

            return q_values.argmax().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("MountainCar-v0")
    model = QNetwork(2, 3)

    episodes = 10000
    min_buffer_size = 1000
    max_buffer_size = 10000
    batch_size = 128

    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 500
    epsilon = epsilon_start

    replay_buffer = []
    epsilon_values = []
    total_rewards = []

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    print("Starting training....")

    epsilon = EPS_START
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() > epsilon:
                with torch.inference_mode():
                    action = torch.argmax(model(state)).item() # max Q
            else:
                action = random.randint(0, env.action_space.n - 1)

            
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_buffer) > BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                
                states, actions, rewards, next_states, dones = batch

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = model(states).gather(1, actions)

                with torch.no_grad():
                    next_q_values = model(next_states).max(1, keepdim=True)[0]
                
                loss = loss_fn(q_values, next_q_values * GAMMA * (1 - dones) + rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * EPS_DECAY, EPS_END)
        print(f"Episode: {episode} | Total Rewards: {total_reward} | Epsilon: {epsilon}")
    
    env.close()
    torch.save(model.state_dict(), "model.pth")

    print("Done.")


if __name__  == "__main__":
    main()