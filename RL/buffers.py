import numpy as np
import torch

from RL.type_aliases import LSTMStates


class BaseBuffer():

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device = "auto",
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

        self.pos = 0
        self.full = False

    def size(self):
        if self.full:
            return self.buffer_size
        return self.pos

    def progress(self):
        return self.size() / self.buffer_size

    def add(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self):
        self.pos = 0
        self.full = False

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds):
        raise NotImplementedError()

    def to_torch(self, array, copy = True):
        if copy:
            return torch.tensor(array, device=self.device)
        return torch.as_tensor(array, device=self.device)

class RolloutBufferSamples():
    def __init__(
        self,
        observations,
        actions,
        values,
        log_probs,
        advantages,
        returns,
    ):
        self.observations = observations
        self.actions = actions
        self.values = values
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns

class RolloutBuffer(BaseBuffer):

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_dim,
        gae_lambda,
        discount,
        n_envs,
        device
    ):
        self.buffer_size = buffer_size
        if isinstance(observation_space, int): # turn observation_space into iterable
            self.observation_space = (observation_space,)
        else:
            self.observation_space = observation_space
        self.action_dim = action_dim
        self.gae_lambda = gae_lambda
        self.discount = discount
        self.n_envs = n_envs
        self.device = device

        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.observation_space), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        if self.full:
            return
        
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action).reshape(self.n_envs, self.action_dim)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size = None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)

        if batch_size is None:  # Return everything, don't create minibatches
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self.get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds].reshape(-1, *self.observation_space),
            self.actions[batch_inds].reshape(-1, self.action_dim),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def compute_discounted_rewards(self):
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_returns = 0
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_returns = self.returns[step + 1]
            self.returns[step] = self.rewards[step] + self.discount * next_values * next_non_terminal

    def compute_return_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        
        # GAE lambda        
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                # if current episode is when episode ended, zero out state value estimate
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.discount * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.discount * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # td lambda
        self.returns = self.advantages + self.values

class RecurrentRolloutBufferSamples(RolloutBufferSamples):
    def __init__(
        self,
        observations,
        actions,
        values,
        log_probs,
        advantages,
        returns,
        lstm_states,
        episode_starts,
    ):
        super().__init__(
            observations,
            actions,
            values,
            log_probs,
            advantages,
            returns
        )
        self.lstm_states = lstm_states
        self.episode_starts = episode_starts

class RecurrentRolloutBuffer(RolloutBuffer):

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_dim,
        hidden_state_shape,
        gae_lambda,
        discount,
        n_envs,
        device
    ):
        self.hidden_state_shape = hidden_state_shape
        super().__init__(
            buffer_size,
            observation_space,
            action_dim,
            gae_lambda,
            discount,
            n_envs,
            device
        )

    def reset(self):
        super().reset()
        self.hidden_states_pi = np.zeros((self.buffer_size, *self.hidden_state_shape), dtype=np.float32)
        self.cell_states_pi = np.zeros((self.buffer_size, *self.hidden_state_shape), dtype=np.float32)
        self.hidden_states_vf = np.zeros((self.buffer_size, *self.hidden_state_shape), dtype=np.float32)
        self.cell_states_vf = np.zeros((self.buffer_size, *self.hidden_state_shape), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        lstm_states: LSTMStates,
        episode_starts: torch.Tensor
    ):
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].detach().cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].detach().cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].detach().cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].detach().cpu().numpy())
        
        super().add(
            obs,
            action,
            reward,
            done,
            value,
            log_prob
        )

    def get(self, sequence_length, sequence_stride, batch_size = None):
        assert self.full, ""
        start_indices = sequence_stride * \
                            np.random.permutation(self.buffer_size // sequence_stride + 1 - int(np.ceil(sequence_length/sequence_stride)))

        if batch_size is None:  # Return everything, don't create minibatches
            batch_size = self.buffer_size

        num_sequences = batch_size // sequence_length
        start_idx = 0
        while start_idx < len(start_indices):
            yield self.get_samples(start_indices[start_idx:start_idx+num_sequences], sequence_length)
            start_idx += num_sequences

    def get_samples(self, batch_inds, sequence_length):
        idxs = np.concatenate([batch_inds + step for step in range(sequence_length)])
        data = (
            self.observations[idxs].reshape(sequence_length, -1, *self.observation_space),
            self.actions[idxs].reshape(-1, self.action_dim),
            self.values[idxs].reshape(-1),
            self.log_probs[idxs].flatten(),
            self.advantages[idxs].flatten(),
            self.returns[idxs].flatten(),
        )
        lstm_states = LSTMStates(
            (self.to_torch(self.reshape_hidden(self.hidden_states_pi[batch_inds])), self.to_torch(self.reshape_hidden(self.cell_states_pi[batch_inds]))),
            (self.to_torch(self.reshape_hidden(self.hidden_states_vf[batch_inds])), self.to_torch(self.reshape_hidden(self.cell_states_vf[batch_inds])))
        )
        episode_starts = self.to_torch(self.episode_starts[idxs].reshape(sequence_length, -1))
        
        return RecurrentRolloutBufferSamples(*tuple(map(self.to_torch, data)),
                                             lstm_states,
                                             episode_starts)

    def reshape_hidden(self, hidden_state):
        # (n_sequences, n_layers, n_envs, hidden_size) -> (n_layers, n_sequences, n_envs, hidden_size)
        # (n_layers, n_sequences, n_envs, hidden_size) -> (n_layers, batch_size, hidden_size)
        n_layers, n_envs, hidden_size = self.hidden_state_shape
        hidden_state = hidden_state.transpose(1,0,2,3)
        hidden_state = hidden_state.reshape(n_layers, -1, hidden_size)
        return hidden_state
        
    

class AggregatedDataset(BaseBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        n_envs = 1,
        device = "auto"
    ):
        self.buffer_size = buffer_size
        if isinstance(observation_space, int): # turn observation_space into iterable
            self.observation_space = (observation_space,)
        else:
            self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.device = device

        self.reset()

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.observation_space), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def _add(self, lst, to_add):
        remaining_space = len(lst) - self.pos
        if not remaining_space < len(to_add):
            lst[self.pos : self.pos+len(to_add)] = to_add
        else:
            lst[self.pos:] = to_add[:remaining_space]
            lst[:len(to_add)-remaining_space] = to_add[remaining_space:]

    def add(self, rollout_buffer):
        self._add(self.observations, rollout_buffer.observations)
        self._add(self.actions, rollout_buffer.actions)
        self._add(self.rewards, rollout_buffer.rewards)
        self._add(self.dones, rollout_buffer.dones)
        self._add(self.values, rollout_buffer.values)
        self._add(self.log_probs, rollout_buffer.log_probs)
        
        self.pos += rollout_buffer.size()
        if self.pos >= self.buffer_size:
            self.full = True
        self.pos %= self.buffer_size

    def get(self, batch_size = None):
        indices = np.random.permutation(self.size())

        if batch_size is None:  # Return everything, don't create minibatches
            batch_size = self.size()

        start_idx = 0
        while start_idx < self.size():
            yield self.get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def get_samples(self, batch_inds):
        data = (
            self.observations[batch_inds].reshape(-1,*self.observation_space),
            self.actions[batch_inds].flatten(),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


















        
