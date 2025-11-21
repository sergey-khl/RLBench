import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import numpy as np
import rlbench
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from rlbench.tasks import ReachTarget, PickAndLift, pick_up_cup
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.basketball_in_hoop import BasketballInHoop
from rlbench.tasks.take_umbrella_out_of_umbrella_stand import TakeUmbrellaOutOfUmbrellaStand

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = MoveArmThenGripper(
    arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
env = Environment(
    action_mode, '', obs_config, False)
env.launch()

def do_obs(obs):
    print(obs.task_low_dim_state)

task = env.get_task(ReachTarget)
num_episodes = 1
demos = task.get_demos(num_episodes, live_demos=True, callable_each_step=do_obs)

observations = []
actions = []
next_observations = []
rewards = []
terminals = []

for demo in demos:
    # Iterate over the fixed episode length to ensure uniform dataset size
    curr_obs = demo[0]
    for next_obs in demo[1:]:

        target_pos = next_obs.task_low_dim_state

        # Observations
        curr_obs_data = np.concatenate([curr_obs.joint_positions, target_pos])
        next_obs_data = np.concatenate([next_obs.joint_positions, target_pos])

        action = np.concatenate([next_obs.joint_velocities, [next_obs.gripper_open]])

        # Reward: Negative Euclidean Distance 
        gripper_pos = next_obs.gripper_pose[:3]
        distance = np.linalg.norm(target_pos - gripper_pos)
        reward = -distance


        # Terminal: True only at the actual end of the expert trajectory
        # terminated = (i >= len(demo) - 1)


        # Record transition
        observations.append(curr_obs_data)
        actions.append(action)
        next_observations.append(next_obs_data)
        rewards.append(reward)
        terminals.append(terminated)

obs_shape = observations[0].shape
act_shape = actions[0].shape

dataset = {
    # Observation Space (Joint Positions)
    'observation_space_low': np.full(obs_shape, -np.inf, dtype=np.float32),
    'observation_space_high': np.full(obs_shape, np.inf, dtype=np.float32),
    'observation_space_shape': np.array(obs_shape, dtype=np.int32),

    # Action Space (Velocity + Gripper)
    'action_space_low': np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,  0], dtype=np.float32),
    'action_space_high': np.array([0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.04], dtype=np.float32),
    'action_space_shape': np.array(act_shape, dtype=np.int32),

    # Data
    'observations': np.array(observations, dtype=np.float32),
    'actions': np.array(actions, dtype=np.float32),
    'next_observations': np.array(next_observations, dtype=np.float32),
    'rewards': np.array(rewards, dtype=np.float32),
    'terminals': np.array(terminals, dtype=bool),
}

np.save('reach_data', dataset, allow_pickle=True)

print('Done')
