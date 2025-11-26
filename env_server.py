from pyrep.objects import Joint
from pyrep.pyrep import time
import zmq
import gymnasium as gym
import numpy as np
import rlbench
from rlbench.action_modes.action_mode import EndEffectorActionMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.basketball_in_hoop import BasketballInHoop
from rlbench.tasks.close_microwave import CloseMicrowave
from rlbench.tasks.reach_target import ReachTarget
from pyquaternion import Quaternion
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 5000, "port for communicating with iql repo")

import signal

running = True

def signal_handler(sig, frame):
    global running
    print("\ninterrupt!")
    running = False

def getTaskData(task):
    task_info = None
    reward = None
    for obj, objtype in task._task._initial_objs_in_scene:
        if obj.get_name() == 'microwave_door': # setup to work with the ReachTarget and CloseMicrowave tasks
            microwave_joint = Joint('microwave_door_joint')
            reward = -microwave_joint.get_joint_position() # in degrees. the more closed the higher the reward
            task_info = obj.get_pose()
        elif obj.get_name() == 'target':
            reward = task._task.reward() # distance based reward
            task_info = obj.get_pose()

    if task_info is None or reward is None:
        raise Exception("data collection only setup to work for reachtarget and closemicrowave tasks")

    return task_info, reward

def run_server(_):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(f"tcp://*:{FLAGS.port}")

    print("RLBench Server running... waiting for IQL client.")

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = EndEffectorActionMode()
    env = Environment(
        action_mode, '', obs_config, headless=False)
    env.launch()

    task = env.get_task(CloseMicrowave)

    socket.setsockopt(zmq.RCVTIMEO, 100)

    curr_obs_data = None

    stats = {
            'episode': {
                 'return': 0,
                 'length': 0
                 }
             }

    # cntrlc stuff
    global running
    signal.signal(signal.SIGINT, signal_handler)
    while running:
        try:
            # Attempt to receive a message
            message = socket.recv_pyobj()
        except zmq.Again:
            # check every 100ms
            continue
        cmd = message['cmd']
        print(cmd)

        if cmd == 'reset':
            obs = task.reset()[1]

            task_info, _ = getTaskData(task)

            curr_obs_data = np.concatenate([obs.gripper_pose, task_info])

            stats = {
                    'episode': {
                         'return': 0,
                         'length': 0
                         }
                    }
                    
            socket.send_pyobj(curr_obs_data)
            
        elif cmd == 'step':
            action = message['action']

            before_step = time.time()

            # normalize the quaternion
            action[3:7] /= np.linalg.norm(action[3:7])
            try:
                obs, _, terminated = task.step(action) # reward from here is not the one we necessarily want to use

                task_info, reward = getTaskData(task)

                curr_obs_data = np.concatenate([obs.gripper_pose, task_info])
            except Exception as e:
                print('failed to step', e)
                # stay in place cus out of bounds or some other problem
                reward = -1 # worst case reward example
                terminated = False

            # print(reward)
                
            stats['episode']['return'] += reward
            stats['episode']['length'] += 1

            after_step = time.time()
            print(after_step-before_step)

            socket.send_pyobj((curr_obs_data, reward, terminated, stats))
            
        elif cmd == 'close':
            running = False
            break

        elif cmd == 'time':
            sent_stamp = message['stamp']
            recieved = time.time()
            diff = recieved - sent_stamp
            socket.send_pyobj(diff)
            break

        elif cmd == 'set_space':
            action_low, action_high = action_mode.action_bounds()
            action_shape = action_mode.action_shape(task._scene)
            obs_shape = np.array([14])
            socket.send_pyobj({
                "observation_space": {
                    'low': np.full(obs_shape, -np.inf, dtype=np.float32),
                    'high': np.full(obs_shape, np.inf, dtype=np.float32),
                    'shape': np.array(obs_shape, dtype=np.int32),
                },
                "action_space": {
                    'low': np.array(action_low, dtype=np.float32),
                    'high': np.array(action_high, dtype=np.float32),
                    'shape': np.array(np.array([action_shape]), dtype=np.int32),
                }
            })

    env.shutdown()
    socket.send_pyobj("Closed")


if __name__ == "__main__":
    app.run(run_server)
