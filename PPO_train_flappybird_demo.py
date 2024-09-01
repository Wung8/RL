# demo of training PPO agent to play flappy bird
# printed score is around 2x the number of pipes the agent passes
# takes around 10 minutes, definitely could train more
# ctrl-c the program and run trainer.test(display=True) to early stop

from environments.flappy_bird import FlappyBirdEnvironment as env
from PPO_agent import PPO


# plots learning curve of agent
# copy paste printed scores into a multi line string as input to plot()
# will update soon (hopefully)
def plot(a):
    lst = []
    for line in a.split('\n'):
        try: lst.append(float(line.replace('#','').split(' ')[-1]))
        except: pass

    temp = lst
    lst = []
    import numpy as np
    for i in range(0,len(temp),20):
        lst.append(np.average(temp[i:i+20]))

    import matplotlib.pyplot as plt
    plt.plot(list(range(len(temp))), temp, alpha=0.3)
    plt.plot([i*20 for i in range(len(lst))],lst)
    plt.show()

if __name__ == "__main__":
    trainer = PPO(env,
                  observation_space=5,
                  action_space=2,
                  n_steps=4_000,
                  n_envs=8)

    trainer.learn(total_steps=200_000)
    print(trainer.test(display=True))
