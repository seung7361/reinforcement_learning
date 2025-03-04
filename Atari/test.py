import gymnasium as gym
import ale_py
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((84, 110)),
    T.CenterCrop((84, 84)),
    T.ToTensor()
])


def process_frame(frame):
    return transform(frame).numpy()

env = gym.make("ALE/Breakout-v5", render_mode="human")
state, _ = env.reset()
env.render()

state = process_frame(state)

plt.imsave('state_image.png', state.transpose(1, 2, 0))