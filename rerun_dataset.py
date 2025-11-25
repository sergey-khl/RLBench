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
from rlbench.tasks.close_microwave import CloseMicrowave
from rlbench.tasks.take_umbrella_out_of_umbrella_stand import TakeUmbrellaOutOfUmbrellaStand
import h5py

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = EndEffectorActionMode()
env = Environment(
    action_mode, '', obs_config, False)
env.launch()

task = env.get_task(CloseMicrowave)

def load_h5py_to_dict(filename):
    loaded_data = {}
    
    with h5py.File(filename, 'r') as f:
        for key, value in f.items():
            print(key, value)
            loaded_data[key] = np.array(value)
            
    return loaded_data

# dataset = np.load("umbrella_data.npy", allow_pickle=True).item()
dataset = load_h5py_to_dict("microwave_data.h5")
print(np.sum(dataset['terminals']))

# Reset to initialize the episode
descriptions, obs = task.reset()

for i, act in enumerate(dataset['actions']):
    obs, reward, terminated = task.step(act)

    if dataset['terminals'][i]:
        task.reset()
        print(f"Episode finished at step {i}")


print('Done')
env.shutdown()
