from environments.cartpole_novelocity import CartPoleEnvironment as env
#from environments.cartpole import CartPoleEnvironment as env
from RL.PPO import PPO

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=2,
                  action_space=3,
                  n_steps=4_096,
                  batch_size=256,
                  discount=.97,
                  ent_coef=1e-3,
                  n_envs=8)

    trainer.learn(total_steps=200_000)
    print(trainer.test(display=True))
