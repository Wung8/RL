import numpy as np
import cv2, math, time, random
import keyboard as k
import torch

class FlappyBirdEnvironment:
    def __init__(self):
        self.gravity = 10
        self.y = 300
        self.x = 200
        self.flap_height = 40
        self.y_vel = 0
        self.dt = 0.5
        self.spacing = 300
        self.pipe_width = 70
        self.pipe_gap = 175
        self.speed = 12

        self.pipes = []
        
        self.reset()

    def reset(self):
        self.y = 300
        self.y_vel = 0
        self.pipe_gap = 500

        self.pipes = [[i*self.spacing + 1000, random.randint(*sorted([self.pipe_gap//2-25,600-self.pipe_gap//2+25])),self.pipe_gap] for i in range(4)]
        #return (self.y/600, self.y_vel/20, *self.scale(self.pipes[0][:2],1/600), *self.scale(self.pipes[1][:2],1/600)), [1,1]
        return self.getInputs(self.pipes[0])

    def scale(self, lst, s):
        return [x*s for x in lst]

    def getInputs(self, closest_pipe):
        return (self.y/600, self.y_vel/20, closest_pipe[0]/600, (closest_pipe[1]-closest_pipe[2]/2)/600, (closest_pipe[1]+closest_pipe[2]/2)/600)
    
    def step(self, action, display=False):
        buffer = 6
        # return (state), reward, done
        if action==1:
            self.y_vel = self.flap_height

        self.y -= self.y_vel * self.dt
        self.y_vel -= self.gravity * self.dt
        if self.y < 50:
            self.y = 50
            self.y_vel = 0

        closest_pipe = self.pipes[0]
        if closest_pipe[0] < 100:
            closest_pipe = self.pipes[1]
            
        if self.y > 600:
            #return (self.y/600, self.y_vel/20, *self.scale(self.pipes[0][:2],1/600), *self.scale(self.pipes[1][:2],1/600)), 0, [1,1], -1
            return self.getInputs(closest_pipe), -1, -1

        if self.pipes[0][0] < -self.pipe_width:
            self.pipes.append([self.pipes[-1][0]+self.spacing, random.randint(*sorted([self.pipe_gap//2-25,600-self.pipe_gap//2+25])),self.pipe_gap])
            del self.pipes[0]
            self.pipe_gap = max(self.pipe_gap-40, 200)
        for pipe in self.pipes:
            pipe[0] -= int(self.speed * self.dt)

        pipe_gap = closest_pipe[2]
        if 200-25-self.pipe_width-buffer < closest_pipe[0] < 200+25+buffer: # hit pipe
            if abs(closest_pipe[1]-self.y) > pipe_gap//2-25+buffer:
                return self.getInputs(closest_pipe), min(-(abs(closest_pipe[1]-self.y) - (pipe_gap//2-25+buffer))/1000, -0.3), -1

        reward = 1.0 if (closest_pipe[0] <= self.x and closest_pipe[0]+int(self.speed*self.dt) > self.x) else 0.01

        if display: self.display()

        # y, y vel, pipe x, pipe y, next pipe x, next pipe y
        #return (self.y/600, self.y_vel/20, *self.scale(self.pipes[0][:2],1/600), *self.scale(self.pipes[1][:2],.1/600)), reward, [1,1], 0
        return self.getInputs(closest_pipe), reward, 0

    def display(self):
        framerate = 60
        size_x = 800
        size_y = 600
        
        img = np.array([[[255,220,140]]],dtype=np.uint8)
        img = img.repeat(size_y,axis=0).repeat(size_x,axis=1)

        closest_pipe = self.pipes[0]
        if closest_pipe[0] < 100:
            closest_pipe = self.pipes[1]
        for pipe in self.pipes:
        #for pipe in [closest_pipe]:
            pipe_gap = pipe[2]
            img = cv2.rectangle(img, (pipe[0],0), (pipe[0]+self.pipe_width, 600), (83,178,85), -1)
            img = cv2.rectangle(img, (pipe[0],pipe[1]+pipe_gap//2), (pipe[0]+self.pipe_width,pipe[1]-pipe_gap//2), (255,220,140), -1)
        
        img = cv2.circle(img,(self.x,int(self.y)),0,(80,255,255),50)
        cv2.imshow('img',img)
        cv2.waitKey(math.ceil(1000/framerate))

    def convState(self, state):
        return torch.tensor([state], dtype=torch.float32)


if __name__ == '__main__':
    env = FlappyBirdEnvironment()
    pressed = False

    for _ in range(1000):
        usr = 0
        if k.is_pressed('space'):
            if pressed == False: usr = 1
            pressed = True
        else: pressed = False
        
        state, reward, truncate = env.step(usr, display=True)
        #if reward: print('ding!')
        if truncate:
            print(reward)
            break
