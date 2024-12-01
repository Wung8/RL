from environments.slime_volleyball_multidiscrete import SlimeEnvironment as env
from RL.IPPO import IPPO
from RL.PPO import PPO

if __name__ == "__main__":
    n_steps = 10_000
    batch_size = 500
    epochs = 10
    ent_coef = 1e-3
    n_envs = 20
    
    left_slime = PPO(env=None,
                     observation_space=8,
                     action_space=(3,2),
                     n_steps = n_steps,
                     batch_size = batch_size,
                     epochs = epochs,
                     discount=.99,
                     ent_coef=ent_coef,
                     n_envs=n_envs)
    
    right_slime = PPO(env=None,
                      observation_space=8,
                      action_space=(3,2),
                      n_steps = n_steps,
                      batch_size = batch_size,
                      epochs = epochs,
                      discount=.99,
                      ent_coef=ent_coef,
                      n_envs=n_envs)
    
    trainer = IPPO(env=env,
                   agents=[left_slime, right_slime],
                   n_envs=n_envs)

    trainer.learn(total_steps=20_000_000)
