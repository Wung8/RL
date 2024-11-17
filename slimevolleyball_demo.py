from environments.slime_volleyball import SlimeEnvironment
from RL.PPO import PPO
import keyboard as k
import numpy as np
import torch
import time

env = SlimeEnvironment()
    
left_slime = PPO(env=None,
                 observation_space=8,
                 action_space=4)

right_slime = PPO(env=None,
                  observation_space=8,
                  action_space=4)

left_slime.model = torch.load("trained_networks\\left_slime_actor.pth")
right_slime.model = torch.load("trained_networks\\right_slime_actor.pth")

def update_usr():
    global usr, u_trigger
    if (k.is_pressed('w') or k.is_pressed('up')) and u_trigger: usr = 3
    elif (k.is_pressed('d') or k.is_pressed('right')): usr = 2
    elif (k.is_pressed('a') or k.is_pressed('left')): usr = 1
    else: usr = 0
    if (k.is_pressed('w') or k.is_pressed('up')): u_trigger = False
    else: u_trigger = True
    

while 1:
    obs = env.reset()
    done = False
    u_trigger = False
    usr = 0
    while not done:
        update_usr()
        with torch.no_grad():
            ai, _, _ = right_slime.get_action(np.array(obs[1], dtype=np.float32))
        obs, r, done = env.step((usr,ai), display=True)
    time.sleep(0.5)
