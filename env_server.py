import zmq
import gymnasium as gym
import numpy as np
import rlbench
from gymnasium.utils.performance import benchmark_step
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape
from rlbench.tasks import ReachTarget, PickAndLift, pick_up_cup
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")

    print("RLBench Server running... waiting for IQL client.")

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete())
    env = Environment(
        action_mode, '', obs_config, False)
    env.launch()

    task = env.get_task(ReachTarget)

    socket.setsockopt(zmq.RCVTIMEO, 100)

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
                obs = task.reset()[0]

                target_pos = obs['task_low_dim_state']

                curr_obs_data = np.concatenate([obs['joint_positions'], target_pos])

                stats = {
                        'episode': {
                             'return': 0,
                             'length': 0
                             }
                        }
                        
                socket.send_pyobj(curr_obs_data)
                
            elif cmd == 'step':
                action = message['action']
                obs, reward, terminated, truncated, info = task.step(action)

                target_pos = obs['task_low_dim_state']

                curr_obs_data = np.concatenate([obs['joint_positions'], target_pos])

                gripper_pos = obs['gripper_pose'][:3]
                distance = np.linalg.norm(target_pos - gripper_pos)
                reward = -distance
                stats['episode']['return'] = reward
                stats['episode']['length'] += 1
                print(stats)

                socket.send_pyobj((curr_obs_data, reward, terminated, truncated, stats))
                
            elif cmd == 'close':
                env.close()
                socket.send_pyobj("Closed")
                break

            elif cmd == 'set_space':
                obs_shape = np.array([10])
                act_shape = np.array([8])
                socket.send_pyobj({
                    "observation_space": {
                        'low': np.full(obs_shape, -np.inf, dtype=np.float32),
                        'high': np.full(obs_shape, np.inf, dtype=np.float32),
                        'shape': np.array(obs_shape, dtype=np.int32),
                    },
                    "action_space": {
                        'low': np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,  0], dtype=np.float32),
                        'high': np.array([0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.04], dtype=np.float32),
                        'shape': np.array(act_shape, dtype=np.int32),
                    }
                })

    # TODO: figure out keyboard interrupt. not workgin 
    except KeyboardInterrupt:
        print("Interrupted...")
    finally:
        # CHANGE 3: Graceful cleanup prevents the QMutex/QObject errors
        print("Cleaning up environment...")
        env.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    run_server()
