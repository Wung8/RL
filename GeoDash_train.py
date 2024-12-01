from environments.geometry_dash import GeoDashEnvironment as env
from RL.PPO import PPO
import winsound, keyboard

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(3,200,400),
                  action_space=2,
                  lr = 3e-4,
                  value_lr = 1e-3,
                  n_steps=4_000,
                  batch_size=250,
                  discount=.99,
                  verbose=False,
                  n_envs=1)

    winsound.Beep(440,500)
    keyboard.wait('k')
    winsound.Beep(840,500)
    trainer.learn(total_steps=20_000_000)
