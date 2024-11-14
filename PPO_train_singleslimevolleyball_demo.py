# demo of training PPO agent to play single player slime volleyball
# printed score is around 0.8x the number of times the slime hit the ball "over" the net

from environments.slime_volleyball_single_player import SlimeEnvironment as env
from RL.PPO import PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=6,
                  action_space=4,
                  lr = 3e-4,
                  value_lr = 1e-3,
                  n_steps=4_000,
                  batch_size=500,
                  discount=.99,
                  n_envs=8)

    trainer.learn(total_steps=1_000_000)
    print(trainer.test(display=True))

