import numpy as np
import torch

class CategoricalDistribution:
    
    def __init__(self, action_space, action_prob):
        self.action_space = action_space
        self.distribution = torch.distributions.Categorical(action_prob)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, actions):
        return self.distribution.log_prob(actions.flatten())

    def entropy(self):
        return self.distribution.entropy()


class MultiCategoricalDistribution:

    def __init__(self, action_space, action_prob):
        self.action_space = action_space
        if len(action_prob.shape) == 1: action_prob = action_prob.reshape(1,-1)
        self.distributions = [torch.distributions.Categorical(split)
                              for split in torch.split(action_prob, action_space, dim=-1)]

    def sample(self):
        return torch.stack([dist.sample() for dist in self.distributions], dim=1)

    def log_prob(self, actions):
        return torch.stack(
            [dist.log_prob(action) for dist,action in zip(self.distributions, torch.unbind(actions,dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self):
        return torch.stack(
            [dist.entropy() for dist in self.distributions], dim=1
        ).sum(dim=1)
