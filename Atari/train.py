import numpy as np
import torch
import ale_py
import cv2
import gymnasium as gym
import torchvision.transforms as T

from constant import *
from ReplayBuffer import ReplayBuffer
from model import QNetwork
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((84, 110)),
    T.CenterCrop((84, 84)),
    T.ToTensor()
])

def process_frame(frame):
    return transform(frame).numpy()

def train():
    print("Starting training...")
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=4, render_mode=None)

    in_dim = (4, 84, 84)
    out_dim = env.action_space.n

    model = QNetwork(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    replay_buffer = ReplayBuffer(min(REPLAY_BUFFER_SIZE, 100000))

    eps = EPS_START
    step = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        state = process_frame(state)
        frame_stack = torch.zeros((4, 84, 84), dtype=torch.float32)
        
        total_reward = 0
        done = False

        while not done:
            eps = max(EPS_END, EPS_START * np.exp(-step / EPS_DECAY))
            step += 1

            frame_stack[:-1] = frame_stack[1:].clone()
            frame_stack[-1] = torch.tensor(state, dtype=torch.float32)
            
            state_tensor = frame_stack.unsqueeze(0).to(device)

            with torch.no_grad():
                action = model.action(state_tensor, eps, device)

            next_state, reward, done, trunc, _ = env.step(action)
            done = done or trunc
            total_reward += reward

            next_state = process_frame(next_state)
            frame_stack[:-1] = frame_stack[1:].clone()
            frame_stack[-1] = torch.tensor(next_state, dtype=torch.float32)

            replay_buffer.push(frame_stack.numpy(), action, reward, frame_stack.numpy(), done)

            if len(replay_buffer) >= BATCH_SIZE:
                batch = replay_buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = batch

                states = torch.tensor(states, dtype=torch.float32, device=device)
                actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

                q_values = model(states).gather(1, actions)
                with torch.no_grad():
                    next_q_values = model(next_states).max(1, keepdim=True)[0]
                target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
                loss = loss_fn(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode: {episode} | Total Rewards: {total_reward} | Epsilon: {eps}")

    env.close()
    torch.save(model.state_dict(), "model.pth")
    print("Done.")

if __name__ == "__main__":
    train()