import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

import scipy, random, time, math, copy
import matplotlib.pyplot as plt

from RL.buffers import RolloutBuffer
from RL.vec_env_handler import ParallelEnvManager

from RL.common_networks import (
    base_MLP_model,
    base_CNN_model,
)


# Note that a smaller epoch size, such as 1-4, is preferred in multi agent environments
class IPPO():

    def __init__(
            self,
            env,
            agents, 
            verbose = True, # use print statements
            n_envs = 1, # vectorized env, how many environments to run in parallel, around #cpus
        ):

        self.env = env
        self.agents = agents
        self.verbose = verbose
        self.n_envs = n_envs

        self.training_history = []
        
        self.env_manager = ParallelEnvManager(self.env, self.n_envs)
        self.last_obs = self.env_manager.reset()

    def set_training_mode(self, training_mode):
        for agent in self.agents: agent.set_training_mode(training_mode)

    def train(self):
        for agent in self.agents: agent.train()

    # main training loop
    def learn(self, total_steps, progress_bar=True):
        num_steps = 0
        while num_steps < total_steps:
            self.collect_rollouts(progress_bar=progress_bar)
            num_steps += self.agents[0].rollout_buffer.size()
            self.train()

            if self.verbose:
                trials = 10
                total_score = [0 for _ in range(len(self.agents)+1)]
                for trial in range(trials):
                    total_score = self.add_lists(total_score, self.test(display=False))
                print(' '+' '.join(f"{x/trials:.3f}" for x in total_score))

    def transpose_info(self, info):
        return [[info[env][agent] for env in range(self.n_envs)] for agent in range(len(self.agents))]

    def add_lists(self, lst_1, lst_2):
        return [lst_1[_]+lst_2[_] for _ in range(len(lst_1))]

    def collect_rollouts(self, progress_bar):
        for agent in self.agents: agent.rollout_buffer.reset()
        
        progress = 0
        if progress_bar:
            print('#',end='')
            
        while not self.agents[0].rollout_buffer.full:
            # convert from (n_envs, n_agents, obs) to (n_agents, n_envs, obs)
            last_obs_info = self.transpose_info(self.last_obs)
            last_obs_info = [np.array(obs, dtype=np.float32) for obs in last_obs_info]
            # action_info = [info_1, info_2, info_3, ...]
            # info = [actions, log_probs, values]
            with torch.no_grad():
                action_info = [
                    self.agents[_].get_action(last_obs_info[_])
                    for _ in range(len(self.agents))
                ]

            # convert from (n_agents, action_info, n_envs) to (n_envs, n_agents)
            actions = [[action_info[agent][0][env] for agent in range(len(self.agents))] for env in range(self.n_envs)]
            # return_info = [next_obs, rewards, dones]
            next_obs_info, rewards_info, dones = self.env_manager.step(actions)
            rewards_info = self.transpose_info(rewards_info)

            # planning on adaptable number of agents, so using len(self.agents) instead
            # of self.num_agents
            for _ in range(len(self.agents)):
                last_obs = last_obs_info[_]
                actions = action_info[_][0]
                rewards = rewards_info[_]
                values = action_info[_][2]
                log_probs = action_info[_][1]
                
                self.agents[_].rollout_buffer.add(
                    last_obs,
                    actions,
                    rewards,
                    dones,
                    values,
                    log_probs,
                )
            
            self.last_obs = next_obs_info

            if progress_bar:
                new_progress = self.agents[0].rollout_buffer.progress()//.1
                if progress < new_progress:
                    print('#',end='')
                    progress = new_progress

        for _ in range(len(self.agents)):
            last_obs_info = self.transpose_info(self.last_obs)
            agent = self.agents[_]
            with torch.no_grad():
                values = agent.get_values(np.array(last_obs_info[_], dtype=np.float32))

            agent.rollout_buffer.compute_return_and_advantage(values, dones)

    def test(self, display, steps=300, **kwargs):
        cumulative_reward = [0 for _ in range(len(self.agents))]
        env = self.env()
        obs_info = env.reset()
        for step in range(steps):
            obs_info = [np.array([obs], dtype=np.float32) for obs in obs_info]
            with torch.no_grad():
                action_info = [
                    self.agents[_].get_action(obs_info[_])
                    for _ in range(len(self.agents))
                ]
            actions = [info[0][0] for info in action_info]

            return_info = env.step(actions, display=display)
            new_obs = return_info[0]
            cumulative_reward = self.add_lists(cumulative_reward, return_info[1])
            done = return_info[2]

            if done: break                    
            obs_info = new_obs

        return cumulative_reward + [sum(cumulative_reward)]


    # training history not implemented yet, fix that first
    def plot_training_history(self, step=20):
        training_history_smoothed = []
        for i in range(0, len(self.training_history), step):
            training_history_smoothed.append(np.average(self.training_history[i:i+20]))

        plt.plot(list(range(len(self.training_history))), self.training_history, alpha=0.3)
        plt.plot([i*20 for i in range(len(training_history_smoothed))], training_history_smoothed)
        plt.show()
        
        
        
        
            

        
        
        

    
        
        
        

