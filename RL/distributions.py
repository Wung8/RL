import numpy as np
import torch

class CategoricalDistribution:
    
    def __init__(self, action_space, action_logits):
        self.action_space = action_space
        self.distribution = torch.distributions.Categorical(logits=action_logits)

    def sample(self):
        return self.distribution.sample()

    def log_prob(self, actions):
        return self.distribution.log_prob(actions.flatten())

    def entropy(self):
        return self.distribution.entropy()


class MultiCategoricalDistribution:

    def __init__(self, action_space, action_logits):
        self.action_space = action_space
        if len(action_logits.shape) == 1: action_logits = action_logits.reshape(1,-1)
        self.distributions = [torch.distributions.Categorical(logits=split)
                              for split in torch.split(action_logits, action_space, dim=-1)]

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
