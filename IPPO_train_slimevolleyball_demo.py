from environments.slime_volleyball import SlimeEnvironment as env
from RL.IPPO import IPPO
from RL.PPO import PPO

if __name__ == "__main__":
    n_steps = 10_000
    batch_size = 500
    epochs = 10
    n_envs = 20
    
    left_slime = PPO(env=None,
                     observation_space=8,
                     action_space=4,
                     n_steps = n_steps,
                     batch_size = batch_size,
                     epochs = epochs,
                     discount=.99,
                     n_envs=n_envs)
    
    right_slime = PPO(env=None,
                      observation_space=8,
                      action_space=4,
                      n_steps = n_steps,
                      batch_size = batch_size,
                      epochs = epochs,
                      discount=.99,
                      n_envs=n_envs)
    
    trainer = IPPO(env=env,
                   agents=[left_slime, right_slime],
                   n_envs=n_envs)

    trainer.learn(total_steps=20_000_000)
