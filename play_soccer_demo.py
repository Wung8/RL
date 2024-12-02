from environments.soccer import SoccerEnv
from RL.IPPO import IPPO
from RL.PPO import PPO
import torch
import numpy as np
import pygame
import time

pygame.init()

env = SoccerEnv()

agents = [PPO(env=None,
              observation_space=47,
              action_space=(3,3,2),
              n_steps=1)
          for i in range(4)]
for i in range(4):
    agents[i].model = torch.load(f"trained_networks\\soccer_models1\\soccer{i}.pt")


p_toggle = False
def update_usr():
    global p_toggle
    actions = [[1,1,0] for i in range(2)]
    keys = pygame.key.get_pressed()

    if keys[pygame.K_p]:
        if not p_toggle:
            time.sleep(0.5)
            while True:
                time.sleep(0.1)
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_p]:
                    break
        p_toggle = True
    else:
        p_toggle = False

    if keys[pygame.K_LEFT]:
        actions[1][0] -= 1
    if keys[pygame.K_RIGHT]:
        actions[1][0] += 1
    if keys[pygame.K_UP]:
        actions[1][1] -= 1
    if keys[pygame.K_DOWN]:
        actions[1][1] += 1
    if keys[pygame.K_PERIOD]:
        actions[1][2] = 1

    if keys[pygame.K_a]:
        actions[0][0] -= 1
    if keys[pygame.K_d]:
        actions[0][0] += 1
    if keys[pygame.K_w]:
        actions[0][1] -= 1
    if keys[pygame.K_s]:
        actions[0][1] += 1
    if keys[pygame.K_t]:
        actions[0][2] = 1

    return actions

while True:
    obs = env.reset()
    done = False
    while not done:
        actions = []
        actions.append(update_usr()[0])
        for i in [1,2,3]:
            with torch.no_grad():
                ai, _, _ = agents[i].get_action(np.array(obs[i], dtype=np.float32))
            actions.append(ai.tolist()[0])
        obs, r, done = env.step(actions, display=True)
    time.sleep(0.5)
