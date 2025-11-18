import zmq
import gymnasium as gym
import numpy as np
import rlbench

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:5555")

    print("RLBench Server running... waiting for IQL client.")

    env = gym.make('rlbench/reach_target-vision-v0', render_mode="human")

    socket.setsockopt(zmq.RCVTIMEO, 100)

    try:
        while True:
            try:
                # Attempt to receive a message
                message = socket.recv_pyobj()
            except zmq.Again:
                # check every 100ms
                continue
            print(message)
            cmd = message['cmd']

            if cmd == 'reset':
                obs = env.reset()[0]
                socket.send_pyobj(obs['joint_positions'])
                
            elif cmd == 'step':
                action = message['action']
                obs, reward, terminated, truncated, info = env.step(action)
                socket.send_pyobj((obs['joint_positions'], reward, terminated, truncated, info))
                
            elif cmd == 'close':
                env.close()
                socket.send_pyobj("Closed")
                break

            elif cmd == 'set_space':
                socket.send_pyobj({
                    "observation_space": {
                        "low": np.array(env.observation_space['joint_positions'].low, dtype=np.float32),
                        "high": np.array(env.observation_space['joint_positions'].high, dtype=np.float32),
                        "shape": np.array(env.observation_space['joint_positions'].shape, dtype=np.float32),
                    },
                    "action_space": {
                        "low": np.array(env.action_space.low, dtype=np.float32),
                        "high": np.array(env.action_space.high, dtype=np.float32),
                        "shape": np.array(env.action_space.shape, dtype=np.float32),
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
