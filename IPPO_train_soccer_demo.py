from environments.soccer import SoccerEnv as env
from RL.IPPO import IPPO
from RL.PPO import PPO

if __name__ == "__main__":
    lr = 1e-4
    value_lr = 3e-4
    n_steps = 10_000
    batch_size = 2_500
    epochs = 5
    discount = .99
    ent_coef = 1e-4
    n_envs = 32
    
    p1 = PPO(env=None,
             observation_space=36,
             action_space=(3,3,2),
             lr = lr,
             value_lr = value_lr,
             n_steps = n_steps,
             batch_size = batch_size,
             epochs = epochs,
             discount = discount,
             ent_coef = ent_coef,
             n_envs=n_envs)
    
    p2 = PPO(env=None,
             observation_space=36,
             action_space=(3,3,2),
             lr = lr,
             value_lr = value_lr,
             n_steps = n_steps,
             batch_size = batch_size,
             epochs = epochs,
             discount = discount,
             ent_coef = ent_coef,
             n_envs=n_envs)

    p3 = PPO(env=None,
             observation_space=36,
             action_space=(3,3,2),
             lr = lr,
             value_lr = value_lr,
             n_steps = n_steps,
             batch_size = batch_size,
             epochs = epochs,
             discount = discount,
             ent_coef = ent_coef,
             n_envs=n_envs)

    p4 = PPO(env=None,
             observation_space=36,
             action_space=(3,3,2),
             lr = lr,
             value_lr = value_lr,
             n_steps = n_steps,
             batch_size = batch_size,
             epochs = epochs,
             discount = discount,
             ent_coef = ent_coef,
             n_envs=n_envs)
    
    trainer = IPPO(env=env,
                   agents=[p1, p2, p3, p4],
                   n_envs=n_envs)

    trainer.learn(total_steps=20_000_000)
