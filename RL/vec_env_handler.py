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
        elif cmd == 'get_expert_actions':
            action = env.get_expert_action()
            pipe.send(action)
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

    def get_expert_actions(self):
        """
        Gets expert actions from all environments.
        Only for Direct Policy Learning

        :return: A list of expert actions for each environment.
        """

        for pipe in self.pipes:
            pipe.send(('get_expert_actions', None))
        
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
