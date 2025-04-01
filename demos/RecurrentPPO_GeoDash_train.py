from environments.geometry_dash import GeoDashEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO
import winsound, keyboard

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=(3,200,400),
                  action_space=2,
                  n_envs=1)

    winsound.Beep(440,500)
    keyboard.wait('k')
    winsound.Beep(840,500)
    trainer.learn(total_steps=20_000_000)
