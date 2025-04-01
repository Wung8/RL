from environments.simple_memory import SimpleMemoryEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=2,
                  action_space=3,
                  n_steps=4_096,
                  n_envs=8)

    trainer.learn(total_steps=200_000)
    print(trainer.test(display=True))
