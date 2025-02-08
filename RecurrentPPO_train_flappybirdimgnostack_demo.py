# demo of training PPO agent to play flappy bird
# printed score is around 2x the number of pipes the agent passes

from environments.flappy_bird_img_nostack import FlappyBirdEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(3,80,80),
                  action_space=2,
                  n_envs=16)

    trainer.learn(total_steps=10_000_000)
    print(trainer.test(display=True))
