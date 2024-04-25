import environments.pendulum as game
from REINFORCE_agent import REINFORCE_agent, REINFORCE_trainer

game.setR(1)

agent = REINFORCE_agent(input_space=4, action_space=3)
trainer = REINFORCE_trainer(game, agent=agent)

trainer.train(epochs=1000, ep_len=1000)
trainer.test(ep_len=1000, display=True)
