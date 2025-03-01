import gymnasium as gym
import random
import torch
from main import QNetwork

env = gym.make("MountainCar-v0", render_mode="human")
state, _ = env.reset()
print(state)

ob = env.observation_space
ac = env.action_space

model = QNetwork(2, 3)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))

state, _ = env.reset()
done = False
while not done:
    action = model(torch.FloatTensor(state)).argmax().item()
    next_state, reward, done, _, _ = env.step(action)

    state = next_state