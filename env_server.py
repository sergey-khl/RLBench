import zmq
import gymnasium as gym
import numpy as np
import rlbench
from rlbench.action_modes.action_mode import EndEffectorActionMode
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks.reach_target import ReachTarget
from pyquaternion import Quaternion

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

    task = env.get_task(ReachTarget)

    socket.setsockopt(zmq.RCVTIMEO, 100)


    curr_obs_data = None

    stats = {
            'episode': {
                 'return': 0,
                 'length': 0
                 }
             }

    try:
        while True:
            try:
                # Attempt to receive a message
                message = socket.recv_pyobj()
            except zmq.Again:
                # check every 100ms
                continue
            cmd = message['cmd']

            if cmd == 'reset':
                obs = task.reset()[1]

                target_pos = obs.task_low_dim_state

                curr_obs_data = np.concatenate([obs.gripper_pose, target_pos])

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

                    target_pos = obs.task_low_dim_state

                    curr_obs_data = np.concatenate([obs.gripper_pose, target_pos])
                except:
                    # stay in place cus out of bounds or some other problem
                    terminated = False

                distance = np.linalg.norm(curr_obs_data[7:] - curr_obs_data[:3])
                reward = -distance
                    
                stats['episode']['return'] = reward
                stats['episode']['length'] += 1
                print(stats)

                socket.send_pyobj((curr_obs_data, reward, terminated, stats))
                
            elif cmd == 'close':
                env.shutdown()
                socket.send_pyobj("Closed")
                break

            elif cmd == 'set_space':
                action_low, action_high = action_mode.action_bounds()
                action_shape = action_mode.action_shape(task._scene)
                obs_shape = np.array([10])
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

    # TODO: figure out keyboard interrupt. not workgin 
    except KeyboardInterrupt:
        print("Interrupted...")
    finally:
        # CHANGE 3: Graceful cleanup prevents the QMutex/QObject errors
        print("Cleaning up environment...")
        env.shutdown()
        socket.close()
        context.term()


if __name__ == "__main__":
    run_server()
