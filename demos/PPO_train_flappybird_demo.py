# demo of training PPO agent to play flappy bird
# printed score is around 2x the number of pipes the agent passes

from environments.flappy_bird import FlappyBirdEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=5,
                  action_space=2,
                  lstm_hidden_size=20,
                  n_steps=4_000,
                  n_envs=8)

    trainer.learn(total_steps=500_000)
    print(trainer.test(display=True))
