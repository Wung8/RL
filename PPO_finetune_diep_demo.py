from environments.diep import DiepioEnvironment as env
from RL.PPO_finetune import PPO_finetune
import torch

model = torch.load("trained_networks\diep_actor.pt")
value_net = torch.load("trained_networks\diep_critic.pt")

if __name__ == "__main__":
    
    trainer = PPO_finetune(env,
                           models=(model, value_net),
                           observation_space=(1,80,80),
                           action_space=32,
                           n_steps=4_000,
                           batch_size=200,
                           epochs=25,
                           n_envs=8,
                           )

    trainer.learn(total_steps=160_000)
    print(trainer.test(display=True))
