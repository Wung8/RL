# demo of training PPO agent to play flappy bird
# printed score is around 2x the number of pipes the agent passes

from environments.flappy_bird_img_nostack import FlappyBirdEnvironment as env
from RL.Recurrent_PPO import Recurrent_PPO as PPO

hyperparams = {
    "lr" = (3e-5, 1e-4, 3e-4),
    "batch_size" = (128, 256, 512),
    "n_envs" = (8, 16, 24),
    "epochs" = (5, 10, 15),
}

learning_curves = {}

def get_hyperparams():
    for lr in hyperparams["lr"]:
        for batch_size in hyperparams["batch_size"]:
            for n_envs in hyperparams["n_envs"]:
                for epochs in hyperparams["epochs"]:
                    yield lr, batch_size, n_envs, epochs

if __name__ == "__main__":
    for lr, batch_size, n_envs, epochs in get_hyperparams():
        for iteration in range(5):
            trainer = PPO(env,
                          observation_space=(3,80,80),
                          action_space=2,
                          lr=lr,
                          value_lr=3*lr,
                          batch_size=batch_size,
                          epochs=epochs,
                          n_envs=n_envs)

            trainer.learn(total_steps=1_000_000)
            if (lr, batch_size, n_envs, epochs) not in learning_curves:
                learning_curves[(lr, batch_size, n_envs, epochs)] = []
            learning_curves[(lr, batch_size, n_envs, epochs)].append(trainer.training_history)
                
