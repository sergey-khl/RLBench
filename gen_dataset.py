import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import numpy as np
import rlbench
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from rlbench.tasks import ReachTarget, PickAndLift, pick_up_cup
from rlbench.action_modes.action_mode import EndEffectorActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.basketball_in_hoop import BasketballInHoop
from rlbench.tasks.take_umbrella_out_of_umbrella_stand import TakeUmbrellaOutOfUmbrellaStand
from pyquaternion import Quaternion

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = EndEffectorActionMode()
env = Environment(
    action_mode, '', obs_config, False)
env.launch()

task = env.get_task(ReachTarget)
num_episodes = 100
demos = task.get_demos(num_episodes, live_demos=True)
env.shutdown()

observations = []
actions = []
next_observations = []
rewards = []
terminals = []


for demo in demos:
    # We iterate up to len-1 because we need pairs (current -> next)
    for i in range(len(demo) - 1):
        curr_obs = demo[i]
        next_obs = demo[i+1]

        target_pos = curr_obs.task_low_dim_state

        curr_obs_data = np.concatenate([curr_obs.gripper_pose, target_pos])
        next_obs_data = np.concatenate([next_obs.gripper_pose, target_pos])
    
        trans_diff = next_obs.gripper_pose[:3] - curr_obs.gripper_pose[:3]

        curr_q_arr = curr_obs.gripper_pose[3:] 
        next_q_arr = next_obs.gripper_pose[3:]

        q_curr = Quaternion(w=curr_q_arr[3], x=curr_q_arr[0], y=curr_q_arr[1], z=curr_q_arr[2])
        q_next = Quaternion(w=next_q_arr[3], x=next_q_arr[0], y=next_q_arr[1], z=next_q_arr[2])
        rot_diff = q_next * q_curr.inverse

        w_d, x_d, y_d, z_d = list(rot_diff)

        action = np.concatenate([trans_diff, [x_d, y_d, z_d, w_d], [next_obs.gripper_open]])

        # Reward: Negative Distance
        gripper_pos = next_obs.gripper_pose[:3]
        distance = np.linalg.norm(target_pos - gripper_pos)
        reward = -distance
        
        terminated = (i == len(demo) - 2)

        observations.append(curr_obs_data)
        actions.append(action)
        next_observations.append(next_obs_data)
        rewards.append(reward)
        terminals.append(terminated)

obs_shape = observations[0].shape
action_low, action_high = action_mode.action_bounds()
action_shape = action_mode.action_shape(task._scene)

dataset = {
    # Observation Space (Joint Positions)
    'observation_space_low': np.full(obs_shape, -np.inf, dtype=np.float32),
    'observation_space_high': np.full(obs_shape, np.inf, dtype=np.float32),
    'observation_space_shape': np.array(obs_shape, dtype=np.int32),

    # Action Space (Velocity + Gripper)
    'action_space_low': np.array(action_low, dtype=np.float32),
    'action_space_high': np.array(action_high, dtype=np.float32),
    'action_space_shape': np.array(action_shape, dtype=np.int32),

    # Data
    'observations': np.array(observations, dtype=np.float32),
    'actions': np.array(actions, dtype=np.float32),
    'next_observations': np.array(next_observations, dtype=np.float32),
    'rewards': np.array(rewards, dtype=np.float32),
    'terminals': np.array(terminals, dtype=bool),
}
np.save('reach_data', dataset, allow_pickle=True)

print('saved data')
