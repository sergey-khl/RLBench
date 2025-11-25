import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import numpy as np
import rlbench
from rlbench.tasks import ReachTarget, PickAndLift, pick_up_cup
from rlbench.action_modes.action_mode import EndEffectorActionMode, MoveArmThenGripper
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.basketball_in_hoop import BasketballInHoop
from rlbench.tasks.plug_charger_in_power_supply import PlugChargerInPowerSupply
from rlbench.tasks.take_umbrella_out_of_umbrella_stand import TakeUmbrellaOutOfUmbrellaStand
from pyquaternion import Quaternion
import h5py

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = EndEffectorActionMode()
env = Environment(
    action_mode, '', obs_config, False)
env.launch()


task = env.get_task(TakeUmbrellaOutOfUmbrellaStand)
num_episodes = 5000

count = 0

def count_steps(obs):
    global count
    print(count)
    count += 1

num_episodes_per_step = 100
steps = num_episodes//num_episodes_per_step
for i in range(steps):
    task.get_demos(num_episodes_per_step, live_demos=True, callable_each_step=count_steps, from_episode_number=i*num_episodes_per_step)

env.shutdown()

