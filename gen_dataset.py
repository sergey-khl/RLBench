import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import numpy as np
from numpy._core.numeric import dtype
from examples.keyboard_observer import KeyboardObserver
import rlbench

# env = gym.make('rlbench/reach_target-vision-v0', render_mode="rgb_array")
env = gym.make('rlbench/reach_target-vision-v0', render_mode="human")

keyboard_obs = KeyboardObserver()


training_steps = 120
# training_steps = 10
episode_length = 40

observations = []
actions = []
next_observations = []
rewards = []
terminals = []

obs = env.reset()[0]
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()[0]

    # action = env.action_space.sample()
    action = keyboard_obs.get_ee_action()
    next_obs, reward, terminated, truncated, info = env.step(action)

    # record transition
    observations.append(obs['joint_positions'])
    actions.append(action)
    next_observations.append(next_obs['joint_positions'])
    rewards.append(reward)
    terminals.append(terminated or truncated)

    obs = next_obs
    env.render()

dataset = {
    'observation_space_low': np.array(env.observation_space['joint_positions'].low, dtype=np.float32),
    'observation_space_high': np.array(env.observation_space['joint_positions'].high, dtype=np.float32),
    'observation_space_shape': np.array(env.observation_space['joint_positions'].shape, dtype=np.float32),
    'action_space_low': np.array(env.action_space.low, dtype=np.float32),
    'action_space_high': np.array(env.action_space.high, dtype=np.float32),
    'action_space_shape': np.array(env.action_space.shape, dtype=np.float32),
    'observations': np.array(observations, dtype=np.float32),
    'actions': np.array(actions, dtype=np.float32),
    'next_observations': np.array(next_observations, dtype=np.float32),
    'rewards': np.array(rewards, dtype=np.float32),
    'terminals': np.array(terminals, dtype=bool),
}

# np.save('test_data', dataset, allow_pickle=True)


print('Done')


fps = benchmark_step(env, target_duration=10)
print(f"FPS: {fps:.2f}")
env.close()
