import cv2
import numpy as np
import math, time
import keyboard as k
import torch

slots = [10,8,5,2]

def nextFrame(usr,display=False):
    global slots
    if display: print(slots[usr])
    return -1, slots[usr], [1]*len(slots), True

def resetEnv():
    return [1], [1]*len(slots)

def convState(state):
    return torch.tensor([state], dtype=torch.float32)
