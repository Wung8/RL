import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

import scipy, random, time, math

from buffers import RolloutBuffer


'''
required functions in env:
    - resetEnv(), returns [state, valid_actions]
    - nextFrame(action), returns [next_state, r, done]
    - convState(state), return [converted_state]

'''

class REINFORCE_Baseline():

    def __init__(
            self,
            env,
            observation_space, # number of values in input
            action_space, # number of values in output
            ep_len, # length of episode before cutoff
            lr = 1e-3, # learning rate of actor
            value_lr = 4e-3, # learning rate of critic (larger than actor)
            n_steps = 4000, # number of steps to train per batch of games
            gae_lambda = -1, # number step for temporal difference
            discount = .97, # discount rate
            normalize_advantage = True, # normalize advantage (in this case returns)
            ent_coef = 1e-4, # entropy coefficient
            max_grad_norm = 0.5, # max gradient norm when clipping
            verbose = True, # use print statements
            models = None, # default none, can specify model
            device = "auto" # gpu or cpu
        ):

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.ep_len = ep_len
        self.lr = lr
        self.value_lr = value_lr
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if models: self.model, self.value_net = models
        else: # create default model
            self.model = nn.Sequential(
              nn.Linear(self.observation_space,64),
              nn.Mish(),
              nn.Linear(64,64),
              nn.Mish(),
              nn.Linear(64,self.action_space),
            )
            self.value_net = nn.Sequential(
              nn.Linear(self.observation_space,64),
              nn.Mish(),
              nn.Linear(64,64),
              nn.Mish(),
              nn.Linear(64,1),
            )
            
        self.rollout_buffer = RolloutBuffer(buffer_size=n_steps,
                                            observation_space=observation_space,
                                            action_space=action_space,
                                            device=device,
                                            gae_lambda=gae_lambda
                                            )
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.opt_value_net = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, weight_decay=1e-5)

    def set_training_mode(self, training_mode):
        if training_mode:
            self.model.to(self.device)
            self.value_net.to(self.device)
        else:
            self.model.to("cpu")
            self.value_net.to("cpu")

    def train(self):
        self.set_training_mode(True)
        
        # get all data in one go
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            
            values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # normalize advantages
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # policy gradient loss
            policy_loss = -(advantages * log_prob).mean()
            
            # entropy loss for exploration
            entropy_loss = torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss

            # backprop policy
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()

            # value loss using GAE
            value_loss = F.mse_loss(rollout_data.returns, values)

            # backprop value
            self.opt_value_net.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.opt_value_net.step()
            
        self.set_training_mode(False)

    # forward actor and critic
    def forward(self, obs):
        actions = F.softmax(self.model(obs), dim=-1)
        values = self.value_net(obs)
        return actions, values

    def get_action(self, obs):
        action_prob, value = self.forward(torch.from_numpy(obs))
        distribution = torch.distributions.Categorical(action_prob)
        action = distribution.sample()
        return action, distribution.log_prob(action), value

    def evaluate_actions(self, observations, actions):
        action_prob, values = self.forward(observations)
        distribution = torch.distributions.Categorical(action_prob)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    # main training loop
    def learn(self, total_steps, progress_bar=True):
        num_steps = 0
        while num_steps < total_steps:
            self.collect_rollouts(progress_bar=progress_bar)
            num_steps += self.rollout_buffer.size()
            self.train()

            if self.verbose:
                trials = 20
                total_score = sum([self.test(ep_len=self.ep_len, display=False) for i in range(trials)])
                print(round(total_score/trials,3))

    def collect_rollouts(self, progress_bar):
        self.rollout_buffer.reset()
        while not self.rollout_buffer.full:
            self.run_episode()
            
    def run_episode(self):
        rewards = []
        values = []
        obs = self.env.resetEnv()
        for step in range(self.ep_len):
            obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                action, log_prob, value = self.get_action(obs)
            new_obs, reward, done = self.env.nextFrame(action.item())
            
            self.rollout_buffer.add(
                obs,
                action,
                reward,
                done,
                value,
                log_prob,
            )

            rewards.append(reward)
            values.append(value)
            
            if done: break
            obs = new_obs

        self.rollout_buffer.compute_discounted_rewards(rewards)

    def test(self, ep_len, display):
        cumulative_reward = 0
        obs = self.env.resetEnv()
        for step in range(ep_len):
            obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                action, action_log_prob, _ = self.get_action(obs)
            new_obs, reward, done = self.env.nextFrame(action.item(), display=display)
            cumulative_reward += reward
            
            if done: break
            obs = new_obs

        return cumulative_reward
        
        
        
            

        
        
        

    
        
        
        

