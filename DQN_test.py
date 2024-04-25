import pendulum as game
from DQN_agent import DQN_agent, DQN_trainer

def plot(a):
    lst = []
    for line in a.split('\n'):
        if "test score:" in line: lst.append(float(line.split(' ')[-1]))

    import matplotlib.pyplot as plt
    plt.plot([i for i in range(len(lst))],lst)
    plt.show()


agent = DQN_agent(input_space=4, action_space=3)
trainer = DQN_trainer(game, agent=agent)
trainer.epsilon_scheduler.reset()

trainer.train(epochs=5000, ep_len=1000)
trainer.test(ep_len=1000, display=True)

