from environments.diep import DiepioEnvironment as env
from RL.PPO import PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(4,80,80),
                  action_space=32,
                  n_steps=4_000,
                  batch_size=250,
                  discount=.99,
                  ent_coef=1e-3,
                  n_envs=8)

    trainer.learn(total_steps=1_000_000)
    print(trainer.test(display=True))
