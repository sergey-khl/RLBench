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

import signal

running = True

def signal_handler(sig, frame):
    global running
    print("\ninterrupt!")
    running = False


def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")

    print("RLBench Server running... waiting for IQL client.")

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = EndEffectorActionMode()
    env = Environment(
        action_mode, '', obs_config, False)
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

        if cmd == 'reset':
            obs = task.reset()[1]


            graspable_objects = task._task.get_graspable_objects()
            umbrella_obj = graspable_objects[0]

            curr_obs_data = np.concatenate([obs.gripper_pose, umbrella_obj.get_pose()])

            stats = {
                    'episode': {
                         'return': 0,
                         'length': 0
                         }
                    }
                    
            socket.send_pyobj(curr_obs_data)
            
        elif cmd == 'step':
            action = message['action']

            # normalize the quaternion
            action[3:7] /= np.linalg.norm(action[3:7])
            try:
                obs, reward, terminated = task.step(action)

                graspable_objects = task._task.get_graspable_objects()
                umbrella_obj = graspable_objects[0]

                curr_obs_data = np.concatenate([obs.gripper_pose, umbrella_obj.get_pose()])
            except:
                # stay in place cus out of bounds or some other problem
                reward = 0
                terminated = False

            print(reward)
                
            stats['episode']['return'] += reward
            stats['episode']['length'] += 1

            # TODO: fix reward
            socket.send_pyobj((curr_obs_data, reward, terminated, stats))
            
        elif cmd == 'close':
            running = False
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
    run_server()
