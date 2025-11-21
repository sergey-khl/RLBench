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

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = EndEffectorActionMode()
env = Environment(
    action_mode, '', obs_config, False)
env.launch()

task = env.get_task(ReachTarget)

dataset = np.load("reach_data.npy", allow_pickle=True).item()

# Reset to initialize the episode
descriptions, obs = task.reset()

for i, act in enumerate(dataset['actions']):
    # act = np.array([-0.2, 0.6, 1.6, 0, 0, 0, 1, 1])
    # Step the environment with the RECORDED action
    # If the logic is correct, the robot should follow the exact path of the demo
    obs, reward, terminated = task.step(act)

    if terminated:
        print(f"Episode finished at step {i}")
        break


print('Done')
