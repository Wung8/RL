from environments.T_maze import TMazeEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(3,3,3),
                  action_space=4,
                  n_steps=4_096,
                  batch_size=1_024,
                  n_envs=16)

    trainer.learn(total_steps=10_000_000)
    print(trainer.test(display=True))
