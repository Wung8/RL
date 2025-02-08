import numpy as np
import pygame
import cv2, math, time, random
import keyboard as k
import torch

SCALE = 50
MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,0,1,1],
    [1,1,1,1,1,1,1,1,1,1,0,1,1],
    [1,1,1,0,0,0,1,1,1,1,0,1,1],
    [1,1,1,0,0,0,0,0,0,0,0,1,1],
    [1,1,1,0,0,0,1,1,1,1,0,1,1],
    [1,1,1,1,1,1,1,1,1,1,0,1,1],
    [1,1,1,1,1,1,1,1,1,1,0,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1],
]
LOCS = {
    "spawn":(4,6),
    "hint":(4,5),
    "door1":(10,3),
    "door2":(10,9),
}

class TMazeEnvironment:
    def __init__(self):
        self.screen_size = 13, 13
        self.surface = pygame.Surface(self.screen_size, pygame.SRCALPHA) 
        self.screen = pygame.display.set_mode(np.multiply(self.screen_size, SCALE), flags=pygame.HIDDEN)
        self.clock = pygame.time.Clock()

        self.colors = {
            "wall":(80,80,80),
            "empty":(40,40,40),
            "player":(170,170,170),
            "blue":(80,120,180),
            "green":(50,150,50),
        }

        self.display_prev = False
        self.reset()

    def reset(self):
        self.player_pos = LOCS["spawn"]
        self.hint = random.choice(["blue", "green"])
        #self.hint = "blue"
        self.door1, self.door2 = [["blue", "green"], ["green", "blue"]][random.randint(0,1)]
        return self.get_inputs()

    def get_inputs(self):
        m = np.array(MAP,dtype=np.uint8)
        m[*LOCS["hint"][::-1]] = [2,3][self.hint == "green"]
        m[*LOCS["door1"][::-1]] = [2,3][self.door1 == "green"]
        m[*LOCS["door2"][::-1]] = [2,3][self.door2 == "green"]
        m = m[self.player_pos[1]-1:self.player_pos[1]+2, self.player_pos[0]-1:self.player_pos[0]+2]
        m = m.tolist()
        for i in range(3):
            for j in range(3):
                if m[j][i] == 0: m[j][i] = self.colors["empty"]
                if m[j][i] == 1: m[j][i] = self.colors["wall"]
                if m[j][i] == 2: m[j][i] = self.colors["blue"]
                if m[j][i] == 3: m[j][i] = self.colors["green"]
        m = np.array(m, dtype=np.float32)/255
        m = np.moveaxis(m, -1, 0)
        return m

        
    def step(self, actions, display=False):
        if self.display_prev != display:
            self.screen = pygame.display.set_mode(np.multiply(self.screen_size, SCALE), flags=[pygame.HIDDEN, pygame.SHOWN][display])
        if actions == -1:
            if display: self.display()
            return 0,0,0
        if actions == 0 and MAP[self.player_pos[1]-1][self.player_pos[0]] != 1:
            self.player_pos = self.player_pos[0], self.player_pos[1]-1
        if actions == 1 and MAP[self.player_pos[1]][self.player_pos[0]+1] != 1:
            self.player_pos = self.player_pos[0]+1, self.player_pos[1]
        if actions == 2 and MAP[self.player_pos[1]+1][self.player_pos[0]] != 1:
            self.player_pos = self.player_pos[0], self.player_pos[1]+1
        if actions == 3 and MAP[self.player_pos[1]][self.player_pos[0]-1] != 1:
            self.player_pos = self.player_pos[0]-1, self.player_pos[1]

        if display: self.display()
        
        if self.player_pos == LOCS["door1"]:
            if self.door1 == self.hint: return self.get_inputs(), 1, 1
            else: return self.get_inputs(), -1, 1
        if self.player_pos == LOCS["door2"]:
            if self.door2 == self.hint: return self.get_inputs(), 1, 1
            else: return self.get_inputs(), -1, 1
        return self.get_inputs(), -0.01, 0
    
    def display(self):
        pygame.event.pump()
        self.surface.fill(self.colors["empty"])

        for i in range(self.screen_size[0]):
            for j in range(self.screen_size[1]):
                if MAP[j][i] == 1:
                    self.surface.set_at((i,j), self.colors["wall"])

        self.surface.set_at(LOCS["hint"], self.colors[self.hint])
        self.surface.set_at(LOCS["door1"], self.colors[self.door1])
        self.surface.set_at(LOCS["door2"], self.colors[self.door2])
        self.surface.set_at(self.player_pos, self.colors["player"])
        
        pygame.transform.scale(self.surface, (np.multiply(self.screen_size, SCALE)), self.screen)
        pygame.display.flip()
        self.clock.tick(10)
        
def get_user_actions():
    actions = -1
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        actions = 0
    if keys[pygame.K_DOWN]:
        actions = 2
    if keys[pygame.K_LEFT]:
        actions = 3
    if keys[pygame.K_RIGHT]:
        actions = 1
    return actions

if __name__ == '__main__':
    pygame.init()
    env = TMazeEnvironment()
    env.reset()
    print('start')
    while True:
        actions = get_user_actions()
        obs, reward, done = env.step(actions, display=1)
        if type(obs) != type(0):
            for o in obs: print(o)
            print(reward)
        if done==1:
            env.reset()
