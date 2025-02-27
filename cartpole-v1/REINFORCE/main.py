import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from typing import List, Tuple

class Policy(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.trajectory = []
        self.log_probs = []

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_dim)
        )


    def reset(self):
        self.trajectory = []
        self.log_probs = []


    def act(self, state):
        # state: (4,) or (B, 4)
        if type(state) is Tuple:
            state, _ = state
        
        state = torch.from_numpy(state).float()
        action = self.model(state) # (2,) or (B, 2)

        action_probs = torch.nn.functional.softmax(action, dim=-1)
        action_probs_np = action_probs.detach().numpy()
        action = np.random.choice(len(action_probs_np), p=action_probs_np)
        self.log_probs.append(torch.log(action_probs[action]))

        return action
    

def train(model, optimizer, gamma=0.99):
    R = 0
    rewards = []

    for reward in model.trajectory[::-1]:
        reward = reward[2]

        R = reward + R * gamma
        rewards.append(R)
    rewards = rewards[::-1]

    loss = 0.
    for r, log_p in zip(rewards, model.log_probs):
        loss += -log_p * r

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()


def main():
    env = gym.make("CartPole-v1")

    in_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    model = Policy(in_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    total_reward = []

    for episode in range(1000):
        model.reset()
        state, _ = env.reset()

        print(f"Episode: {episode}")
        for _ in range(10000): # max trajectory length
            action = model.act(state)
            state, reward, terminated, _, _ = env.step(action)
            model.trajectory.append((state, action, reward))
            env.render()

            if terminated:
                break

        train(model, optimizer)
        total_reward.append(sum([x[2] for x in model.trajectory]))

        print(f"Total reward: {total_reward[-1]}")
        if total_reward[-1] == 10000.0:
            break
    
    env.close()

    torch.save(model.state_dict(), "model.pth")

    plt.plot(total_reward, label="total rewards")
    plt.xlabel("episode")
    plt.ylabel("Total Reward")

    plt.savefig("reward.png")


if __name__ == "__main__":
    main()