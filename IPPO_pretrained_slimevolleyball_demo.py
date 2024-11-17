from environments.slime_volleyball import SlimeEnvironment as env
from RL.IPPO import IPPO
from RL.PPO import PPO
import torch

if __name__ == "__main__":
    
    left_slime = PPO(env=None,
                     observation_space=8,
                     action_space=4)
    
    right_slime = PPO(env=None,
                      observation_space=8,
                      action_space=4)
    
    trainer = IPPO(env=env,
                   agents=[left_slime, right_slime])

    left_slime.model = torch.load("trained_networks\\left_slime_actor.pth")
    right_slime.model = torch.load("trained_networks\\right_slime_actor.pth")

    while 1:
        trainer.test(display=True, n_steps=9999)
