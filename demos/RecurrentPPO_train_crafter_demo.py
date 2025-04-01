# demo of training PPO agent to play flappy bird
# printed score is around 2x the number of pipes the agent passes

from environments.crafter.crafter import CrafterEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO
#from RL.PPO import PPO as PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(3,72,96),
                  action_space=(5,2,2,2,2,2),
                  batch_size=256,
                  n_envs=16)

    trainer.learn(total_steps=10_000_000)
    print(trainer.test(display=True))
