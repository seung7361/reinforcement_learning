import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="human")
state, _ = env.reset()
env.render()

print(env.action_space)
print(env.observation_space)

# while True:
#     state, reward, term, _, _ = env.step(env.action_space.sample())

#     print(state, state.shape)

#     test = state.reshape(-1)
#     print([a for a in test if a != 0])

#     if term:
#         break