import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode="human")
in_dim = env.observation_space.shape[0]
out_dim = env.action_space.n

print(in_dim, out_dim)

state = env.reset()
for _ in range(5000):
    env.render()

    if type(state) is tuple:
        state, _ = state

    print(state)
    action = random.choice([0, 1])
    state, reward, term, _, _ = env.step(action=action)

    print(f"action={action}, state={state}, reward={reward}, term={term}")

    if term:
        break