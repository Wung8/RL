import cv2
import numpy as np
import math, random, time
import keyboard as k
import torch


pos = 0
size = 10
cutoff = 50
board = [random.randint(0,1) for i in range(size)]

def displayEnv(pos):
    print(pos)

def nextFrame(usr,display=False):
    global pos, count
    count += 1
    pos = max(pos + [-1,1][usr==board[pos]], 0)
    if display: displayEnv(pos)
    if pos == len(board) or count == cutoff: return [1 if i==pos else 0 for i in range(10)], 10, [1,1], True
    return [1 if i==pos else 0 for i in range(size)], 0, [1,1], False
    
def resetEnv():
    global pos, count
    count = 0
    pos = 0
    return [1 if i==pos else 0 for i in range(size)], [1,1]

def convState(state):
    return torch.tensor([state], dtype=torch.float32)

total_r = 0
ep_len = 500

resetEnv()
if __name__ == "__main__":
    #while True:
    for _ in range(ep_len):
        usr = int(input()=='d')
        state, r, valid_actions, truncate = nextFrame(usr,display=True)
        if truncate: break
