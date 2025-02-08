import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

import scipy, random, time, math
import matplotlib.pyplot as plt

from RL.buffers import RolloutBuffer, AggregatedDataset
from RL.vec_env_handler import ParallelEnvManager

from RL.common_networks import (
    base_MLP_model,
    base_CNN_model,
)


class LinearBetaScheduler():
    def __init__(self, rampdown_rounds):
        self.rampdown_rounds = rampdown_rounds
        self.round_num = -1

    def __call__(self):
        self.round_num += 1
        return min(1, max(0, (self.rampdown_rounds - self.round_num) / self.rampdown_rounds))


class ExponentialBetaScheduler():
    def __init__(self, decay):
        self.beta = 1/decay
        self.decay = decay

    def __call__(self):
        self.beta *= decay
        return self.beta
    

class DAgger():

    def __init__(
            self,
            env,
            observation_space, # number of values in input
            action_space, # number of values in output
            lr = 1e-4, # learning rate of actor
            value_lr = 3e-4, # learning rate of critic (larger than actor)
            n_steps = 4000, # number of steps to train per batch of games
            batch_size = 128, # minibatch size
            epochs = 10, # number of epochs 
            discount = .97, # discount rate
            gae_lambda = 0.9, # td lambda for GAE
            normalize_advantage = True, # normalize advantage (in this case returns)
            ent_coef = 1e-4, # entropy coefficient
            max_grad_norm = 0.5, # max gradient norm when clipping
            verbose = True, # use print statements
            models = None, # default none, can specify model
            n_envs = 1, # vectorized env, how many environments to run in parallel
            buffer_size = 1_000_000, # max size of aggregated dataset, FIFO
            device = "auto" # gpu or cpu
        ):

        if isinstance(observation_space, int):
            observation_space = (observation_space,)

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.value_lr = value_lr
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.device = device

        self.training_history = []

        if models: self.model, self.value_net = models
        else: # create default model
            if len(observation_space) == 1:
                self.model = base_MLP_model(input_space=observation_space, output_space=action_space)
                self.value_net = base_MLP_model(input_space=observation_space, output_space=1)
            else:
                self.model = base_CNN_model(input_space=observation_space, output_space=action_space)
                self.value_net = base_CNN_model(input_space=observation_space, output_space=1)
            
        self.rollout_buffer = RolloutBuffer(buffer_size=n_steps,
                                            observation_space=observation_space,
                                            action_space=action_space,
                                            device=device,
                                            gae_lambda=gae_lambda,
                                            discount=discount,
                                            n_envs=n_envs,
                                            )
        self.dataset = AggregatedDataset(buffer_size=buffer_size,
                                         observation_space=observation_space,
                                         action_space=action_space,
                                         device=device,
                                         n_envs=n_envs,
                                         )
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.opt_value_net = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, weight_decay=1e-5)

        self.env_manager = ParallelEnvManager(self.env, self.n_envs)
        self.last_obs = self.env_manager.reset()

    def set_training_mode(self, training_mode):
        if training_mode:
            self.model.to(self.device)
            self.value_net.to(self.device)
        else:
            self.model.to("cpu")
            self.value_net.to("cpu")        

    def train(self):
        self.set_training_mode(True)

        for epoch in range(self.epochs):       
            for rollout_data in self.dataset.get(self.batch_size):
                expert_actions = rollout_data.actions

                logits = self.model(rollout_data.observations)
                values = self.value_net(rollout_data.observations)
                values = values.flatten()

                loss = F.cross_entropy(logits, expert_actions.to(torch.int64))

                # backprop policy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()

                # value loss using GAE
                value_loss = F.mse_loss(values, rollout_data.returns)

                # backprop value
                self.opt_value_net.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.opt_value_net.step()
            
        self.set_training_mode(False)

    # forward actor and critic
    def forward(self, obs):
        actions = self.model(obs)
        values = self.value_net(obs)
        return actions, values

    def get_action(self, obs):
        action_logits, value = self.forward(torch.from_numpy(obs))
        distribution = torch.distributions.Categorical(action_logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def get_values(self, obs):
        return self.value_net(torch.from_numpy(obs))

    def learn(self, total_steps, beta_scheduler=None, progress_bar=True):
        if beta_scheduler == None:
            beta_scheduler = LinearBetaScheduler(total_steps//self.n_steps)
            
        num_steps = 0
        while num_steps < total_steps:
            self.beta = beta_scheduler()
            self.collect_rollouts(progress_bar=progress_bar)
            num_steps += self.rollout_buffer.size()
            self.dataset.add(self.rollout_buffer)
            self.train()

            if self.verbose:
                trials = 10
                total_score = sum([self.test(display=False) for i in range(trials)])
                avg_score = total_score / trials
                print(round(total_score/trials,3))

                self.training_history.append(avg_score)
                

    def collect_rollouts(self, progress_bar):
        self.rollout_buffer.reset()
        
        progress = 0
        if progress_bar:
            print('#',end='')
            
        while not self.rollout_buffer.full:
            self.last_obs = np.array(self.last_obs, dtype=np.float32)
            
            with torch.no_grad():
                actions, log_probs, values = self.get_action(self.last_obs)
            expert_actions = np.array(self.env_manager.get_expert_actions(), dtype=np.float32)

            if random.random() < self.beta:
                new_obs, rewards, dones = self.env_manager.step(np.array(expert_actions))
            else:
                new_obs, rewards, dones = self.env_manager.step(np.array(actions))
            
            self.rollout_buffer.add(
                self.last_obs,
                expert_actions,
                rewards,
                dones,
                values,
                log_probs,
            )
            
            self.last_obs = new_obs

            if progress_bar:
                new_progress = self.rollout_buffer.progress()//.1
                if progress < new_progress:
                    print('#',end='')
                    progress = new_progress

        self.last_obs = np.array(self.last_obs, dtype=np.float32)
        with torch.no_grad():
            values = self.get_values(self.last_obs)

        self.rollout_buffer.compute_return_and_advantage(values, dones)

    def test(self, display, n_steps=300, **kwargs):
        cumulative_reward = 0
        env = self.env()
        obs = env.reset()
        for step in range(n_steps):
            obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                action, action_log_prob, _ = self.get_action(obs)
            new_obs, reward, done = env.step(action.item(), display=display, **kwargs)
            cumulative_reward += reward
            
            if done: break
            obs = new_obs

        return cumulative_reward

    def plot_training_history(self, step=20):
        training_history_smoothed = []
        for i in range(0, len(self.training_history), step):
            training_history_smoothed.append(np.average(self.training_history[i:i+20]))

        plt.plot(list(range(len(self.training_history))), self.training_history, alpha=0.3)
        plt.plot([i*20 for i in range(len(training_history_smoothed))], training_history_smoothed)
        plt.show()
        
        
