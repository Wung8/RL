import multiprocessing as mp
import numpy as np
import sys

def env_worker(pipe, env):
    """
    Worker function to run the environment loop.
    
    :param pipe: The communication pipe (connection) to the main process.
    :param env: An environment instance.
    """
    env = env()
    while True:
        cmd, action = pipe.recv()
        if cmd == 'step':
            obs, reward, done = env.step(action)
            if done:
                obs = env.reset()
            pipe.send((obs, reward, done))
        elif cmd == 'reset':
            obs = env.reset()
            pipe.send(obs)
        elif cmd == 'close':
            #env.close()
            pipe.close()
            break

class ParallelEnvManager:
    def __init__(self, env_callable, num_envs):
        """
        Initialize the Parallel Environment Manager.
        
        :param env_callable: A callable that returns a new instance of the environment.
        :param num_envs: Number of environments to run in parallel.
        """
        self.num_envs = num_envs
        self.pipes = []
        self.processes = []
        
        for _ in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            process = mp.Process(target=env_worker, args=(child_conn, env_callable))
            process.start()
            self.pipes.append(parent_conn)
            self.processes.append(process)

    def step(self, actions):
        """
        Step all environments with the provided actions.
        
        :param actions: A list of actions, one for each environment.
        :return: A tuple of lists: (next_observations, rewards, dones, infos)
        """
        for pipe, action in zip(self.pipes, actions):
            pipe.send(('step', action))
        
        results = [pipe.recv() for pipe in self.pipes]
        next_obs, rewards, dones = zip(*results)
        next_obs = np.array(next_obs, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return next_obs, rewards, dones

    def reset(self):
        """
        Reset all environments.
        
        :return: A list of initial observations for each environment.
        """
        for pipe in self.pipes:
            pipe.send(('reset', None))
        
        results = [pipe.recv() for pipe in self.pipes]
        return list(results)

    def close(self):
        """
        Close all environments and processes.
        """
        for pipe in self.pipes:
            pipe.send(('close', None))
        for process in self.processes:
            process.join()

# Example usage
if __name__ == "__main__":
    from environments.cartpole import CartPoleEnvironment as env
    import random

    num_envs = 2
    env_manager = ParallelEnvManager(env_callable=env, num_envs=num_envs)

    # Reset all environments
    observations = env_manager.reset()
    print("Initial Observations:", observations)

    # Example of taking a step in all environments
    actions = [random.randrange(0,3) for i in range(num_envs)]
    next_obs, rewards, dones = env_manager.step(actions)
    print("Next Observations:", next_obs)
    print("Rewards:", rewards)
    print("Dones:", dones)

    # Close environments
    env_manager.close()
