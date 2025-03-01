import numpy as np
import torch
import gymnasium as gym

import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class QNetwork(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim)
        )

    def forward(self, state):
        if type(state) is tuple:
            state, _ = state
        
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        return self.net(state)
    

def main():
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

    for episode in range(episodes):
        state, _ = env.reset()

        total_reward = 0
        done = False
        epsilon_values.append(epsilon)

        while not done:
            if random.random() > epsilon:
                with torch.inference_mode():
                    action = torch.argmax(model(state)).item() # max Q
            else:
                action = random.randint(0, env.action_space.n - 1)

            
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            total_reward += reward

            if len(replay_buffer) > min_buffer_size:
                states, actions, rewards, next_states, dones = \
                    zip(*random.sample(replay_buffer, batch_size))
                
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = model(states)
                next_q_values = model(next_states).max(1).values

                target = (1 - dones) * next_q_values * gamma + reward
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = loss_fn(q_value, target)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

            while len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)
        print(f"Episode: {episode}, total reward: {total_reward}, epsilon: {epsilon}")

        total_rewards.append(total_reward)

    print("Training done.")
    torch.save(model.state_dict(), "./model.pth")

    plt.plot(epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.savefig('epsilon_decay.png')
    plt.close()

    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.savefig('total_rewards.png')
    plt.close()


if __name__ == "__main__":
    main()