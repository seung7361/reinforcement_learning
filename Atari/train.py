import numpy as np
import torch
import ale_py
import cv2
import gymnasium as gym

gym.register_envs(ale_py)

import random

from constant import *
from ReplayBuffer import ReplayBuffer
from model import QNetwork
from util import process_frame
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print("Starting training...")

    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=3)

    in_dim = (4, 84, 84)
    out_dim = env.action_space.n

    model = QNetwork(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    eps = EPS_START
    step = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        state = process_frame(state)
        frame_stack = deque([state] * 4, maxlen=4) # (4, 84, 84)

        total_reward = 0
        done = False

        while not done:
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)
            step += 1

            state = np.stack(frame_stack, axis=0) # (4, 84, 84)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device) # (1, 4, 84, 84)

            with torch.no_grad():
                action = model.action(state_tensor, eps, device)

            next_state, reward, done, trunc, _ = env.step(action)
            done = done or trunc
            total_reward += reward

            next_state = process_frame(next_state)
            frame_stack.append(next_state)
            next_state_stack = np.stack(frame_stack, axis=0) # (4, 84, 84)

            replay_buffer.push(state, action, reward, next_state_stack, done)


            if len(replay_buffer) >= 1000:
                batch = replay_buffer.sample(BATCH_SIZE)

                states, actions, rewards, next_states, dones = batch

                states = torch.FloatTensor(states).reshape(BATCH_SIZE, 4, 84, 84).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).reshape(BATCH_SIZE, 4, 84, 84).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = model(states).gather(1, actions)

                with torch.no_grad():
                    next_q_values = model(next_states).max(1, keepdim=True)[0]
                
                loss = loss_fn(q_values, next_q_values * GAMMA * (1 - dones) + rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode: {episode} | Total Rewards: {total_reward} | Epsilon: {eps}")

    env.close()
    torch.save(model.state_dict(), "model.pth")

    print("Done.")


if __name__ == "__main__":
    train()