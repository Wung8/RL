# demo of training PPO agent to play single player slime volleyball
# printed score is around 0.8x the number of times the slime hit the ball "over" the net
# takes around 20 minutes, definitely could train more
# ctrl-c the program and run trainer.test(display=True) to early stop

from environments.slime_volleyball_single_player import SlimeEnvironment as env
from PPO_agent import PPO

# plots learning curve
# copy pase print statements into multi-line string as input to plot()
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
                  observation_space=6,
                  action_space=4,
                  lr = 3e-4,
                  value_lr = 1e-3,
                  n_steps=4_000,
                  batch_size=500,
                  discount=.99,
                  n_envs=8)

    trainer.learn(total_steps=1_000_000)
    print(trainer.test(display=True))

