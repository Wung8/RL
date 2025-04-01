# demo of trained PPO agent playing single player slime volleyball

from environments.slime_volleyball_single_player import SlimeEnvironment as env
from RL.PPO import PPO
import torch


if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=6,
                  action_space=4,
                  n_steps=4_000,
                  batch_size=500,
                  discount=.99,
                  n_envs=8)

    trainer.model.load_state_dict(torch.load(r"trained_networks\single_slime.pt"))
    while 1:        
        print(trainer.test(display=True, n_steps=9999))
