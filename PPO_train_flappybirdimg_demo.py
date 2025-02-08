# demo of training PPO agent to play flappy bird
# printed score is around 2x the number of pipes the agent passes

from environments.flappy_bird_img import FlappyBirdEnvironment as env
from RL.PPO import PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(4,80,80),
                  action_space=2,
                  n_steps=4_000,
                  n_envs=12)

    trainer.learn(total_steps=50_000_000)
    print(trainer.test(display=True))
