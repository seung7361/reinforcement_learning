import gymnasium as gym
import ale_py
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from model import QNetwork

gym.register_envs(ale_py)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((84, 110)),
    T.CenterCrop((84, 84)),
    T.ToTensor()
])


def process_frame(frame):
    return transform(frame).numpy()


in_dim = (4, 84, 84)
out_dim = 4

model = QNetwork(in_dim, out_dim)
model.load_state_dict(torch.load("model.pth"))
model.eval()

env = gym.make("ALE/Breakout-v5", render_mode="human", obs_type="grayscale")
state, _ = env.reset()
env.render()

done = False
frame_stack = torch.zeros((4, 84, 84), dtype=torch.float32)

while not done:
    state = process_frame(state)
    frame_stack[:-1] = frame_stack[1:].clone()
    frame_stack[-1] = torch.tensor(state, dtype=torch.float32)

    state_tensor = frame_stack.unsqueeze(0)
    action = model(state_tensor).argmax().item()

    state, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.render()

env.close()