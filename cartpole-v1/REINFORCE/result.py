import gymnasium as gym
import torch

from main import Policy

model = Policy(4, 2)
model.load_state_dict(torch.load("model.pth"))

def main():
    env = gym.make("CartPole-v1", render_mode="human")

    state, _ = env.reset()

    for _ in range(10000):
        action = model.act(state)
        state, _, term, _, _ = env.step(action)
        env.render()

        if term:
            break
    
    env.close()

if __name__ == "__main__":
    main()