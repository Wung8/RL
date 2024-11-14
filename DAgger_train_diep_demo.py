from environments.diep import DiepioEnvironment as env
from RL.DAgger import DAgger

if __name__ == "__main__":
    
    trainer = DAgger(env,
                     observation_space=(1,80,80),
                     action_space=32,
                     n_steps=4_000,
                     batch_size=200,
                     epochs=25,
                     n_envs=8,
                     buffer_size=200_000
                     )

    trainer.learn(total_steps=200_000)
    print(trainer.test(display=True))
