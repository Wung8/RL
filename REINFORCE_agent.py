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

def default_model(input_space, output_space):
    model = nn.Sequential(
      nn.Linear(input_space,64),
      nn.Mish(),
      nn.Linear(64,64),
      nn.Mish(),
      nn.Linear(64,output_space)
    )
    return model

class REINFORCE():

    def __init__(
            self,
            env,
            observation_space, # number of values in input
            action_space, # number of values in output
            lr = 1e-3, # learning rate
            n_steps = 4000, # number of steps to train per batch of games
            discount = .97, # discount rate
            normalize_advantage = True, # normalize advantage (in this case returns)
            ent_coef = 1e-4, # entropy coefficient
            max_grad_norm = 0.5, # max gradient norm when clipping
            verbose = True, # use print statements
            model = None, # default none, can specify model
            n_envs = 1, # vectorized env, how many environments to run in parallel, around #cpus
            device = "auto" # gpu or cpu
        ):

        self.env = env
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.n_steps = n_steps
        self.discount = discount
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model: self.model = model
        else: # create default model
            self.model = default_model(input_space=observation_space, output_space=action_space)
            
        self.rollout_buffer = RolloutBuffer(buffer_size=n_steps,
                                            observation_space=observation_space,
                                            action_space=action_space,
                                            device=device,
                                            gae_lambda=gae_lambda,
                                            discount=discount,
                                            n_envs=n_envs,
                                            )
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        self.env_manager = ParallelEnvManager(self.env, self.n_envs)
        self.last_obs = self.env_manager.reset()
    
    def set_training_mode(self, training_mode):
        if training_mode:
            self.model.to(self.device)
        else:
            self.model.to("cpu")
            

    def train(self):
        self.set_training_mode(True)
        
        # get all data in one go
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            returns = rollout_data.returns 
            log_prob, entropy = self.evaluate_actions(rollout_data.observations, actions)

            # normalize returns
            if self.normalize_advantage:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # policy gradient loss
            policy_loss = -(returns * log_prob).mean()
            
            # entropy loss for exploration
            entropy_loss = torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss

            # backprop
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            
        self.set_training_mode(False)

    # forward pass of actor
    def forward(self, obs):
        actions = F.softmax(self.model(obs), dim=-1)
        return actions

    def get_action(self, obs):
        action_prob = self.forward(torch.from_numpy(obs))
        distribution = torch.distributions.Categorical(action_prob)
        action = distribution.sample()
        return action, distribution.log_prob(action)

    def evaluate_actions(self, observations, actions):
        action_prob = self.forward(observations)
        distribution = torch.distributions.Categorical(action_prob)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_prob, entropy

    # main training loop
    def learn(self, total_steps, progress_bar=True):
        num_steps = 0
        while num_steps < total_steps:
            self.collect_rollouts(progress_bar=progress_bar)
            num_steps += self.rollout_buffer.size()
            self.train()

            if self.verbose:
                trials = 20
                total_score = sum([self.test(display=False) for i in range(trials)])
                print(round(total_score/trials,3))

    def collect_rollouts(self, progress_bar):
        self.rollout_buffer.reset()
        
        progress = 0
        if progress_bar:
            print('#',end='')
            
        while not self.rollout_buffer.full:
            
            self.last_obs = np.array(self.last_obs, dtype=np.float32)
            with torch.no_grad():
                actions, log_probs = self.get_action(self.last_obs)
            new_obs, rewards, dones = self.env_manager.step(np.array(actions))
            
            self.rollout_buffer.add(
                self.last_obs,
                actions,
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

        self.rollout_buffer.compute_discounted_rewards(rewards)

    def test(self, display):
        cumulative_reward = 0
        env = self.env()
        obs = env.reset()
        for step in range(100_000):
            obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                action, action_log_prob = self.get_action(obs)
            new_obs, reward, done = env.step(action.item(), display=display)
            cumulative_reward += reward
            
            if done: break
            obs = new_obs

        return cumulative_reward
        
        
        
            

        
        
        

    
        
        
        

