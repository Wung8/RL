from environments.cartpole import CartPoleEnvironment as env
from RL.DAgger import DAgger

if __name__ == "__main__":
    
    trainer = DAgger(env,
                     observation_space=4,
                     action_space=3,
                     n_steps=4_000,
                     n_envs=8)

    trainer.learn(total_steps=20_000)
    print(trainer.test(display=True))
