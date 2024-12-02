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
    agents[i].model = torch.load(f"trained_networks\\soccer_models\\soccer{i}.pt")


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

num_players = int(input("how many humans? 1-2 "))
player_idxs = []

print('''
player ids:
|0   2|
|1   3|
''')

for i in range(num_players):
    if i == 0:
        print("p1 controls: move=wasd, kick=t")
    else:
        print("p2 controls: move=arrow_keys, kick=.")
    usr = int(input("player id? 0-3 "))
    player_idxs.append(usr)

while True:
    ai_idxs = [i for i in range(4) if i not in player_idxs]
    obs = env.reset()
    done = False
    while not done:
        actions = []
        for i in ai_idxs:
            with torch.no_grad():
                ai, _, _ = agents[i].get_action(np.array(obs[i], dtype=np.float32))
            actions.append(ai.tolist()[0])
        user_actions = update_usr()
        for i in range(len(player_idxs)):
            actions.insert(player_idxs[i], user_actions[i])
        obs, r, done = env.step(actions, display=True)
    time.sleep(0.5)
