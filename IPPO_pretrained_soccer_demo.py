from environments.soccer import SoccerEnv as env
from RL.IPPO import IPPO
from RL.PPO import PPO
import torch

if __name__ == "__main__":
    
    agents = [PPO(env=None,
                  observation_space=36,
                  action_space=(3,3,2))
              for i in range(4)]
    for i in range(4):
        agents[i].model = torch.load(f"trained_networks\\soccer_models1\\soccer{i}.pt")
    
    trainer = IPPO(env=env,
                   agents=agents)

    while 1:
        trainer.test(display=True, n_steps=9999)
