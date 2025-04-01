import cv2 # only necessary for environments with image observation and using wrapper
import numpy as np
import gymnasium as gym
from RL.env_utils import *


'''
default configuration
'''
env_name, obs_space, action_space, algo, pixel_obs, skipframe = "N/A", -1, -1, "PPO", False, False

'''
different environments and their different configurations. uncomment the wanted configuration
and comment out the rest.
'''
env_name, obs_space, action_space = "CartPole-v1", 4, 2
#env_name, obs_space, action_space = "Acrobot-v1", 6, 3
#env_name, obs_space, action_space, algo = "LunarLander-v2", 8, 4, "Recurrent_PPO"
#env_name, obs_space, action_space, algo, pixel_obs = "CarRacing-v2", (3, 64, 64), 5, "Recurrent_PPO", True
#env_name, obs_space, action_space, algo, pixel_obs, skipframe = "ALE/Breakout-v5", (3, 64, 64), 4, "Recurrent_PPO", True, True
#env_name, obs_space, action_space, algo, pixel_obs, skipframe = "ALE/SpaceInvaders-v5", (3, 64, 64), 6, "PPO", True, True


'''
env generator function to pass into trainer, used for creating multiple instances to vectorize.
'''
def env(**kwargs):
    try:
        env = gym.make(env_name, continuous=False, **kwargs)
    except:
        env = gym.make(env_name, **kwargs)

    if skipframe: env = SkipFrameWrapper(env)
    if pixel_obs: env = PixelObsWrapper(env)
        
    return env
        
        
if __name__ == "__main__":
    '''
    import either PPO or Recurrent_PPO under the name PPO.
    '''
    if algo == "PPO":
        from RL.PPO import PPO
    if algo == "Recurrent_PPO":
        from RL.Recurrent_PPO import Recurrent_PPO as PPO

    '''
    n_envs is the number of instances of the environment running in parallel. this means if n_envs=12
    and total_steps=500_000, the agent will collect 500_000*12 frames to train on. this allows more
    quantity and diversity in the data from the environment with less affect on runtime (unless you
    overload your CPU).
    '''
    agent = PPO(
        env,
        observation_space=obs_space,
        action_space=action_space,
        n_envs=12
    )

    '''
    trains the agent. total_steps is the number of frames collected from environment before ending
    training, simpler environments tend to only need around 300_000 steps, while more complicated
    environments (especially image observation ones) need 1_000_000 to 10_000_000 steps.

    feel free to interrupt the training at any point during data collection (when the '#' are being
    printed or score was just printed) but not during network learning (when all 10 '#'s have been
    printed and score has not been printed), and continue training with agent.learn(...).
    '''
    #agent.learn(total_steps=300_000)
    agent.learn(total_steps=5_000_000)
    '''
    gym environments need the argument 'render_mode="human"' to display the environment as an image.
    agent.test returns the test run score, which is printed out.
    '''
    print(agent.test(render_mode="human"))





