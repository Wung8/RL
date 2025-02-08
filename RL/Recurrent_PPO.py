from copy import deepcopy

import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F

import scipy, math
import matplotlib.pyplot as plt

from RL.PPO import PPO

from RL.buffers import RecurrentRolloutBuffer
from RL.vec_env_handler import ParallelEnvManager

from RL.type_aliases import LSTMStates
from RL.common_networks import (
    base_LSTM_model
)
from RL.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)


class Recurrent_PPO(PPO):

    def __init__(
            self,
            env, # callable that creates an instance of the env, None for no env
            observation_space, # number of values in input
            action_space, # number of values in output
            lstm_hidden_size = 256, # number of hidden units in LSTM
            n_lstm_layers = 1, # number of LSTM layers 
            lr = 1e-4, # learning rate of actor
            value_lr = 3e-4, # learning rate of critic (larger than actor)
            n_steps = 4096, # number of steps to train per batch of games
            batch_size = 256, # minibatch size
            sequence_length = 16, # sequence length for truncated backpropagation through time
            sequence_stride = 8, # timesteps between start of each truncated sequence
            epochs = 10, # number of epochs 
            clip_range = 0.2, # clip range for PPO
            discount = .97, # discount rate
            gae_lambda = 0.9, # td lambda for GAE
            normalize_advantage = True, # normalize advantage (in this case returns)
            ent_coef = 1e-4, # entropy coefficient
            max_grad_norm = 0.5, # max gradient norm when clipping
            verbose = True, # use print statements
            models = None, # default none, can specify model
            n_envs = 1, # vectorized env, how many environments to run in parallel, around #cpus
            device = "auto" # gpu or cpu
        ):

        super().__init__(
            env=env,
            observation_space=observation_space,
            action_space=action_space,
            lr=lr,
            value_lr=value_lr,
            n_steps=n_steps,
            batch_size=batch_size,
            epochs=epochs,
            clip_range=clip_range,
            discount=discount,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            verbose=verbose,
            models=models,
            n_envs=n_envs,
            device=device
        )

        #assert batch_size % sequence_stride == 0, "batch_size must be divisible by sequence_stride"
        assert sequence_length > sequence_stride, "sequence stride should be less than sequence_length"

        if isinstance(observation_space, int):
            observation_space = (observation_space,)
        if isinstance(action_space, int):
            action_space = (action_space,)
        if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

        if models: self.model, self.value_net = models
        else: # create default model
            self.model = base_LSTM_model(input_space=observation_space,
                                         output_space=action_space,
                                         n_lstm_layers = n_lstm_layers,
                                         hidden_size=lstm_hidden_size)
            self.value_net = base_LSTM_model(input_space=observation_space,
                                             output_space=(1,),
                                             n_lstm_layers = n_lstm_layers,
                                             hidden_size=lstm_hidden_size)

        self.hidden_state_shape = n_lstm_layers, self.n_envs, lstm_hidden_size
        self.last_lstm_states = LSTMStates(
            (
                torch.zeros(self.hidden_state_shape, dtype=torch.float32),
                torch.zeros(self.hidden_state_shape, dtype=torch.float32),
            ),
            (
                torch.zeros(self.hidden_state_shape, dtype=torch.float32),
                torch.zeros(self.hidden_state_shape, dtype=torch.float32),
            )
        )
            
        self.rollout_buffer = RecurrentRolloutBuffer(
            buffer_size=n_steps,
            observation_space=observation_space,
            action_dim=self.action_dim,
            hidden_state_shape = self.hidden_state_shape,
            device=device,
            gae_lambda=gae_lambda,
            discount=discount,
            n_envs=n_envs,
        )
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.opt_value_net = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, weight_decay=1e-5)

        self.last_episode_starts = torch.zeros(n_envs, dtype=torch.float32)

    def train(self):
        self.set_training_mode(True)
        clip_range = self.clip_range

        for epoch in range(self.epochs):       
            for rollout_data in self.rollout_buffer.get(self.sequence_length, self.sequence_stride, self.batch_size):
                actions = rollout_data.actions
                
                values, log_prob, entropy = self.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()

                # normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.log_probs)

                # clipped loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
                policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2))
                
                # entropy loss for exploration
                entropy_loss = torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss

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

    def process_sequence(
            self,
            observations,
            lstm_states,
            episode_starts,
            network
        ):
        # observations : shape (seq_length, batch_size, observation_shape)
        # episode_starts : shape (seq_length, batch_size)

        # if we don't have to reset state in middle of sequence we can
        # input entire sequence, which speeds up things
        if torch.all(episode_starts == 0.0):
            # shape (seq_length, n_sequences, output_shape), (batch_size, output_shape)
            lstm_output, lstm_states = network.forward(observations, lstm_states)
            lstm_output = torch.flatten(lstm_output, start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        n_seq = lstm_states[0].shape[1]
        lstm_output = []
        for obs, start in zip(observations, episode_starts):
            out, lstm_states = network.forward(
                obs.unsqueeze(0),
                (
                    (1-episode_starts).view(1, n_seq, 1) * lstm_states[0],
                    (1-episode_starts).view(1, n_seq, 1) * lstm_states[1],
                )
            )
            lstm_output += [out]

        # (seq_length, n_sequences, output_shape) -> (batch_size, output_shape)
        lstm_output = torch.flatten(torch.cat(lstm_output), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def forward(self, obs, lstm_states, episode_starts):
        if len(obs.shape) == len(self.observation_space)+1: obs = obs.unsqueeze(0)
        actions, lstm_states_pi = self.process_sequence(obs, lstm_states.pi, episode_starts, self.model)
        values, lstm_states_vf = self.process_sequence(obs, lstm_states.vf, episode_starts, self.value_net)
        return actions, values, LSTMStates(lstm_states_pi, lstm_states_vf)

    def get_action(self, obs, lstm_states, episode_starts):
        action_logits, value, lstm_states = self.forward(torch.from_numpy(obs), lstm_states, episode_starts)
        distribution = self.distribution(self.action_space, action_logits)
        action = distribution.sample()
        return action, distribution.log_prob(action), value, lstm_states

    def get_values(self, obs, lstm_states, episode_starts):
        return self.process_sequence(obs, lstm_states, episode_starts, self.value_net)[0]

    def evaluate_actions(self, observations, actions, lstm_states, episode_starts):
        action_logits, values, _ = self.forward(observations, lstm_states, episode_starts)
        distribution = self.distribution(self.action_space, action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def collect_rollouts(self, progress_bar):
        self.rollout_buffer.reset()
        
        progress = 0
        if progress_bar:
            print('#',end='')

        lstm_states = deepcopy(self.last_lstm_states)
        total_rewards, total_dones = 0, 0
        
        while not self.rollout_buffer.full:
            self.last_obs = np.array(self.last_obs, dtype=np.float32)

            with torch.no_grad():
                episode_starts = torch.tensor(self.last_episode_starts, dtype=torch.float32)
                actions, log_probs, values, lstm_states = self.get_action(self.last_obs, lstm_states, episode_starts)
            new_obs, rewards, dones = self.env_manager.step(np.array(actions))
            
            self.rollout_buffer.add(
                self.last_obs,
                actions,
                rewards,
                dones,
                values,
                log_probs,
                self.last_lstm_states,
                self.last_episode_starts
            )
            total_rewards += sum(rewards)
            total_dones += abs(sum(dones))
            
            self.last_obs = new_obs
            self.last_lstm_states = lstm_states
            self.last_episode_starts = dones

            if progress_bar:
                new_progress = self.rollout_buffer.progress()//.1
                if progress < new_progress:
                    print('#',end='')
                    progress = new_progress

        self.last_obs = np.array(self.last_obs, dtype=np.float32)
        with torch.no_grad():
            values = self.get_values(torch.from_numpy(self.last_obs).unsqueeze(0), lstm_states.vf, episode_starts)

        self.rollout_buffer.compute_return_and_advantage(values, dones)
        return total_rewards/max(total_dones,1)

    def test(self, display, n_steps=300, **kwargs):
        cumulative_reward = 0
        env = self.env()
        obs = env.reset()
        hidden_state_shape = list(self.hidden_state_shape)
        hidden_state_shape[1] = 1
        lstm_states = LSTMStates(
            (
                torch.zeros(hidden_state_shape, dtype=torch.float32),
                torch.zeros(hidden_state_shape, dtype=torch.float32),
            ),
            (
                torch.zeros(hidden_state_shape, dtype=torch.float32),
                torch.zeros(hidden_state_shape, dtype=torch.float32),
            )
        )
        episode_starts = [[0]]
        for step in range(n_steps):
            obs = np.array(obs, dtype=np.float32)
            with torch.no_grad():
                episode_starts = torch.tensor(episode_starts, dtype=torch.float32)
                action, log_probs, values, lstm_states = self.get_action(np.expand_dims(obs,0), lstm_states, episode_starts)
            if self.action_dim == 1: action = action.item()
            else: action = action.tolist()[0]
            new_obs, reward, done = env.step(action, display=display, **kwargs)
            cumulative_reward += reward
            
            if done: break
            obs = new_obs

        return cumulative_reward

        
        
        
            

        
        
        

    
        
        
        

