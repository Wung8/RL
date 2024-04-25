import cv2
import numpy as np
import math, time
import keyboard as k
import torch

pi = math.pi
framerate = 20
friction = 0.97
g = 10
speed = 5

# keeping pendulum up
def r1(x, theta): return 5*((-math.sin(theta)+1)/2)**2# - 2*max(0,abs(x)-.7)
# keeping pendulum down
def r2(x, theta): return (abs(1-abs((theta%(2*pi)-pi/2)/pi)))**.5 - 2*abs(x)

def setR(i):
    global rfun
    rfun = [r1,r2][i]

def displayEnv(pos,theta,t):
    pos = (int(pos[0]),int(pos[1]))
    size = 800
    l = 100
    start,end = pos, (int(pos[0]+l*math.cos(theta)),int(pos[1]+l*math.sin(theta)))
    
    img = np.array([[[255]]],dtype=np.uint8)
    img = img.repeat(size,axis=0).repeat(size,axis=1)
    img = cv2.line(img,(200,400),(260,400),(200),2)
    img = cv2.line(img,(600,400),(540,400),(200),2)
    img = cv2.circle(img,(200,400),0,(0),5)
    img = cv2.circle(img,(600,400),0,(0),5)
    img = cv2.line(img,start,end,(0),2)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    #cv2.waitKey(max(1, int(1000/framerate - (time.time()-t))))

def nextFrame(usr,display=False):
    global theta,pos,vel,step
    step += 1
    usr = usr-1
    
    t = time.time()
    if pos[0] == 200:
        if rfun==r1 and usr==-1: return -1, -10, [0,0,0], True
        usr = max(0,usr)
        pos = (200,400)
    if pos[0] == 600:
        if rfun==r1 and usr==1: return -1, -10, [0,0,0], True
        usr = min(0,usr)
        pos = (600,400)
    a = np.dot((usr,0),(math.sin(theta),math.cos(theta)))*speed
    vel += a/framerate
    pos = np.add(pos,(speed*usr,0))

    a = -np.dot((0,-1),(math.sin(theta),math.cos(theta)))*g
    vel += a/framerate
    vel *= friction
    theta += vel/framerate*3
    theta = theta % (2*pi)
    
    if display: displayEnv(pos,theta,t)
    # ( x, angle, vel, t ), r
    return ((pos[0]-400)/200 * 2, theta, vel, step/1000), rfun((pos[0]-400)/200, theta), [1,1,1], False

def resetEnv():
    global theta,pos,vel,step
    if rfun == r1: theta = pi/2
    if rfun == r2: theta = -pi/2
    vel = 0
    pos = (400,400)
    step = 0
    return ((pos[0]-400)/200 * 2, theta, vel, step/1000), [1,1,1]

def convState(state):
    return torch.tensor([state], dtype=torch.float32)

total_r = 0
ep_len = 500

setR(0)
resetEnv()
if __name__ == "__main__":
    #while True:
    for _ in range(ep_len):
        usr = 1
        if k.is_pressed('a') or k.is_pressed('left'): usr = 0
        if k.is_pressed('d') or k.is_pressed('right'): usr = 2
        state, r, truncate = nextFrame(usr,display=True)
        total_r += r
        if _%10 == 0:
            _ = 0
            print(r)
            print(state[2])
        
    print(total_r)
