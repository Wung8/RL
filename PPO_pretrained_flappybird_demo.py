# demo of trained PPO agent playing flappy bird

from environments.flappy_bird import FlappyBirdEnvironment as env
from RL.PPO import PPO
import torch


if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=5,
                  action_space=2,
                  n_steps=4_000,
                  batch_size=500,
                  discount=.99,
                  n_envs=8)

    trainer.model.load_state_dict(torch.load(r"trained_networks\flappy_bird.pt"))
    while 1:        
        print(trainer.test(display=True, n_steps=9999))
