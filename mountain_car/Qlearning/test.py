import gymnasium as gym
import random

env = gym.make("MountainCar-v0", render_mode="human")
state, _ = env.reset()
print(state)

ob = env.observation_space
ac = env.action_space

print(ob,ac, state)
print(random.choice(ac))

for _ in range(10000):
    action = random.randint(0, env.action_space.n - 1)
    state, reward, term, _, _ = env.step(action)

    if term:
        break