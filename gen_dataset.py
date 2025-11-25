import gymnasium as gym
from gymnasium.utils.performance import benchmark_step
import numpy as np
from pyrep.const import PrimitiveShape
import rlbench
from rlbench.tasks import ReachTarget, PickAndLift
from rlbench.action_modes.action_mode import EndEffectorActionMode, MoveArmThenGripper
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.basketball_in_hoop import BasketballInHoop
from rlbench.tasks.close_microwave import CloseMicrowave
from pyrep.objects.joint import Joint
from pyquaternion import Quaternion
import h5py
from pyrep.objects.shape import Shape



# CHANGE ME !!!!!
task_name = "box" # one of box or reach

obs_config = ObservationConfig()
obs_config.set_all(True)

action_mode = EndEffectorActionMode()
env = Environment(
    action_mode, '', obs_config, False)
env.launch()


if task_name == "box":
    task = env.get_task(CloseMicrowave)
    h5_filename = 'box_data.h5'
elif task_name == "reach":
    task = env.get_task(ReachTarget)
    h5_filename = 'reach_data.h5'
else:
    raise Exception("task not implemented")

num_episodes = 80

h5_file_handle = None
h5_datasets = {}
data_index = 0
written_episodes = 0

prev_obs = None
current_episode_data = []

def collectData(curr_obs):
    global prev_obs, current_episode_data

    # keep track of previous observation so we can find action from it 
    success, terminated = task._task.success()
    if prev_obs is None and not terminated:
        prev_obs = curr_obs
        return
    elif prev_obs is None:
        return

    task_info = None
    reward = None
    for obj, objtype in task._task._initial_objs_in_scene:
        if obj.get_name() == 'microwave_door': # setup to work with the ReachTarget and CloseMicrowave tasks
            microwave_joint = Joint('microwave_door_joint')
            reward = -microwave_joint.get_joint_position() # in degrees. the more closed the higher the reward
            task_info = obj.get_pose()
        elif obj.get_name() == 'target':
            reward = task._task.reward()
            task_info = obj.get_pose()

    if task_info is None or reward is None:
        raise Exception("data collection only setup to work for reachtarget and closemicrowave tasks")

    prev_obs_data = np.concatenate([prev_obs.gripper_pose, task_info])
    curr_obs_data = np.concatenate([curr_obs.gripper_pose, task_info])

    trans_diff = curr_obs.gripper_pose[:3] - prev_obs.gripper_pose[:3]

    prev_q_arr = prev_obs.gripper_pose[3:] 
    curr_q_arr = curr_obs.gripper_pose[3:]

    q_prev = Quaternion(w=prev_q_arr[3], x=prev_q_arr[0], y=prev_q_arr[1], z=prev_q_arr[2])
    q_curr = Quaternion(w=curr_q_arr[3], x=curr_q_arr[0], y=curr_q_arr[1], z=curr_q_arr[2])
    rot_diff = q_curr * q_prev.inverse

    w_d, x_d, y_d, z_d = list(rot_diff)

    action = np.concatenate([trans_diff, [x_d, y_d, z_d, w_d], [curr_obs.gripper_open]])

    current_episode_data.append({
        'observations': prev_obs_data,
        'actions': action,
        'next_observations': curr_obs_data,
        'rewards': np.array([reward], dtype=np.float32), # Wrap in array for shape compatibility
        'terminals': np.array([terminated], dtype=bool) # Wrap in array for shape compatibility
    })

    prev_obs = curr_obs

    if terminated and success:
        appendToFile(current_episode_data)

        prev_obs = None
        current_episode_data = []
    elif not success and terminated:
        raise Exception("bad demo")

def appendToFile(episode_data):
    global data_index, written_episodes

    num_steps = len(episode_data)
    if num_steps <= 1:
        return

    written_episodes += 1
    print(f"wrote episode {len(current_episode_data)}, {written_episodes}")

    new_size = h5_datasets['observations'].shape[0] + num_steps
    end_index = data_index + num_steps

    # works cus we specified max size
    for key in h5_datasets.keys():
        h5_datasets[key].resize(new_size, axis=0)

    # fill the dataset with the episode
    for key in h5_datasets.keys():
        data_to_write = np.stack([d[key] for d in episode_data])
        h5_datasets[key][data_index:end_index] = data_to_write

    data_index = end_index

def initH5pyFile(obs_shape, action_shape):
    global h5_file_handle, h5_datasets, num_episodes

    action_low, action_high = action_mode.action_bounds()
    
    h5_file_handle = h5py.File(h5_filename, 'w')

    # shape data
    metadata = {
        'observation_space_low': np.full(obs_shape, -np.inf, dtype=np.float32),
        'observation_space_high': np.full(obs_shape, np.inf, dtype=np.float32),
        'observation_space_shape': np.array(obs_shape, dtype=np.int32),
        'action_space_low': np.array(action_low, dtype=np.float32),
        'action_space_high': np.array(action_high, dtype=np.float32),
        'action_space_shape': np.array((action_shape,), dtype=np.int32),
    }

    for key, array in metadata.items():
        h5_file_handle.create_dataset(key, data=array)

    trajectory_keys = ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
    shape_map = {
        'observations': obs_shape, 'actions': (action_shape, ), 
        'next_observations': obs_shape, 'rewards': (1,), 'terminals': (1,)
    }
    dtype_map = {
        'observations': np.float32, 'actions': np.float32, 
        'next_observations': np.float32, 'rewards': np.float32, 'terminals': bool
    }

    for key in trajectory_keys:
        current_shape = (0,) + shape_map[key]
        max_shape = (num_episodes* 100,) + shape_map[key]
        
        dset = h5_file_handle.create_dataset(
            key, shape=current_shape, maxshape=max_shape, 
            dtype=dtype_map[key], compression="gzip", compression_opts=4
        )
        h5_datasets[key] = dset
        print(f"Initialized HDF5 dataset: {key}")



obs_shape = (14, ) # both tasks this is done for have obs shape 14 to make life easy
action_shape = action_mode.action_shape(task._scene) # size 8 for 7dof joints and 1 for gripper

# add meta data and start rolling file
initH5pyFile(obs_shape, action_shape)

# collect data
demos = task.get_demos(num_episodes, live_demos=True, callable_each_step=collectData)

if h5_file_handle:
    h5_file_handle.close()
    print("file close")

env.shutdown()

print('saved data')
