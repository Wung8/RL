import numpy as np
import cv2, math, time, random
import keyboard as k
import torch


def add(l1, l2): return [a+b for a,b in zip(l1,l2)]
def subtract(l1, l2): return [a-b for a,b in zip(l1,l2)]
def scale(lst, s): return [a*s for a in lst]
def norm(lst): return [i/(sum(map(abs, lst))) for i in lst]
def mag(lst): return math.sqrt(sum(i**2 for i in lst))
def turn_int(lst): return [int(i) for i in lst]

class SlimeEnvironment:
    def __init__(self):
        self.wall_hit = 0
        self.last_frame = time.time()
        self.framerate = 45
        self.dt = 0.1 # in game time passed per frame
        self.skip_frames = 2 # number of physics frames processed between displaying
        self.screen_size = (400,400) # width, height
        self.timestep = 0

        self.ground_level = 320
        self.net_x = 400
        self.net_level = self.ground_level-40
        self.net_width = 5 # width from center, so net is net_width*2 px wide
        self.slime = Slime()
        self.ball = Ball()

        self.colors = {'bg': (245,248,254),
                       'ground': (124,227,150),
                       'net': (241,211,124),
                       'ball': (245,191,0),
                       'left': (243,74,0),
                       'right': (0,147,255)}
        for color in self.colors.keys():
            self.colors[color] = self.colors[color][::-1] # convert rgb to bgr
            

    def reset(self):
        self.timestep = 0
        self.slime.reset()
        self.ball.reset()
        return self.getInputs()

    def scale(self, lst, s):
        return [x*s for x in lst]

    def shift(self, pos, s):
        return (pos[0]+s, pos[1])

    def flip_pos(self, pos):
        return (self.screen_size[0] - pos[0], pos[1])

    def flip_vel(self, vel):
        return (-vel[0], vel[1])

    def getInputs(self, left=True):
        return (*scale(self.slime.pos,1/400),
                *scale(self.ball.pos,1/400),
                *scale(self.ball.vel,1/60))

    def step(self, action, display=False, is_skip=False):
        self.timestep += 1
        ball_side = 2*int(self.ball.pos[0] > self.screen_size[0]/2) - 1
        
        # process skip frames
        if is_skip == False:
            self.wall_hit = 0
            for i in range(self.skip_frames):
                result = self.step(action, display=False, is_skip=True)
                if result == -1:
                    return self.getInputs(), -1, True

        slime = self.slime

        # slime controls
        match action:
            case 0: slime.vel[0] = 0
            case 1: slime.vel[0] = -slime.move_speed
            case 2: slime.vel[0] = slime.move_speed
            case 3:
                if slime.pos[1]==self.ground_level:
                    slime.vel[1] = -slime.jump_height

        slime.pos = add(slime.pos, scale(slime.vel, self.dt))
        slime.vel = add(slime.vel, scale([0,slime.gravity], self.dt))

        # keep slime in bounds
        if slime.pos[0] < slime.radius:
            slime.pos[0] = slime.radius
        if slime.pos[0] > self.net_x - self.net_width//2:
            slime.pos[0] = self.net_x - self.net_width//2

        # if slime on floor
        if slime.pos[1] >= self.ground_level:
            slime.pos[1] = self.ground_level
            slime.vel[1] = 0

        # detect collisions with slimes
        slime_hit = False # overrides game end

        # collision is when bottom of slime is below the ball and the slime and ball are close enough together
        if slime.pos[1] > self.ball.pos[1] and math.dist(slime.pos, self.ball.pos) < slime.radius + self.ball.radius:
            slime_hit = True
            angle = norm(subtract(self.ball.pos, slime.pos))
            new_vel = add(scale(angle, self.ball.bounce_vel), slime.vel)
            if new_vel[0] > self.ball.max_vel_x: new_vel[0] = self.ball.max_vel_x
            if new_vel[0] < -self.ball.max_vel_x: new_vel[0] = -self.ball.max_vel_x
            if new_vel[1] < -self.ball.max_vel_y: new_vel[1] = -self.ball.max_vel_y
            self.ball.vel = new_vel

        # all physics values are scaled by dt
        self.ball.pos = add(self.ball.pos, scale(self.ball.vel, self.dt)) # apply velocity
        self.ball.vel = add(self.ball.vel, scale([0,self.ball.gravity], self.dt)) # apply gravity, + is down

        # bounce off walls
        if self.ball.pos[0]-self.ball.radius <= 0:
            self.ball.vel[0] = -self.ball.vel[0]
            self.ball.pos[0] = 0 + self.ball.radius + 1
        if self.ball.pos[0] >= self.screen_size[0]:
            self.wall_hit = 1
            self.ball.vel[0] = -self.ball.vel[0]
            self.ball.pos[0] = self.screen_size[0] - self.ball.radius - 1

        # bounce off net
        if abs(self.ball.pos[0]-self.net_x) <= self.net_width+self.ball.radius and self.ball.pos[1] >= self.net_level-self.ball.radius:
            # if ball is going up then it hit the side
            if self.ball.vel[0] > 0: side = self.net_x - self.net_width
            else: side = self.net_x + self.net_width
            if self.ball.vel[1] < 0:
                self.ball.vel[0] = -self.ball.vel[0]
                self.ball.pos[0] = side + [-1,1][side>self.net_x] * self.ball.radius
            else:
                iy = self.ball.radius-abs(self.net_level - self.ball.pos[1])
                ix = self.ball.radius-abs(side - self.ball.pos[0])
                dx, dy = self.ball.vel
                if abs(ix/dx) < abs(iy/dy):
                    self.ball.vel[0] = -self.ball.vel[0]
                    self.ball.pos[0] = side + [-1,1][side>self.net_x] * self.ball.radius
                else:
                    self.ball.vel[1] = -self.ball.vel[1]
                    self.ball.pos[1] = self.net_level - self.ball.radius - 1

        if display: self.display()
            
        # if touching ground
        if not slime_hit and self.ball.pos[1]+self.ball.radius > self.ground_level:
            if is_skip: return -1
            else: return self.getInputs(), -1, True

        # penalize ball being on side
        return self.getInputs(), -.0001 + self.wall_hit, False

    def display(self):
        # fill in bg
        img = np.array([[self.colors['bg']]], dtype=np.uint8)
        img = img.repeat(self.screen_size[1],axis=0).repeat(self.screen_size[0],axis=1)

        # draw net
        img = cv2.rectangle(img,
                           (self.net_x-self.net_width, self.net_level),
                           (self.net_x+self.net_width, self.ground_level),
                           self.colors['net'], thickness=-1)

        # fill in ground
        img = cv2.rectangle(img,
                            (0, self.ground_level),
                            self.screen_size,
                            self.colors['ground'], thickness=-1)

        # draw slimes as half circles
        slime = self.slime
        img = cv2.ellipse(img,
                          turn_int(slime.pos),
                          [slime.radius, slime.radius],
                          0, 180, 360,
                          self.colors['left'], thickness=-1)

        # draw ball
        img = cv2.circle(img, turn_int(self.ball.pos), self.ball.radius, self.colors['ball'], thickness=-1)

        cv2.imshow('img', img)
        cv2.waitKey(max(int(1000/self.framerate-(time.time()-self.last_frame)), 1))
        self.last_frame = time.time()

    def convState(self, state):
        return torch.tensor([state], dtype=torch.float32)
    

class Slime:
    def __init__(self):
        self.gravity = 10
        self.radius = 30
        self.pos = [0,0]
        self.vel = [0,0]
        self.jump_height = 40
        self.move_speed = 30
        
        self.reset()

    def reset(self):
        self.pos = [200,400]
        self.vel = [0,0]


class Ball:
    def __init__(self):
        self.gravity = 10
        self.radius = 15
        self.pos = [0,0]
        self.vel = [0,0]
        self.bounce_vel = 50
        self.max_vel_x = 75
        self.max_vel_y = 50
        
        self.reset()

    def reset(self):
        self.pos = [200,200]
        self.vel = [0,0]


def update_usr():
    global usr1, u1p
    if k.is_pressed('w') and u1p: usr1 = 3
    elif k.is_pressed('d'): usr1 = 2
    elif k.is_pressed('a'): usr1 = 1
    else: usr1 = 0
    if k.is_pressed('w'): u1p = False
    else: u1p = True


if __name__ == '__main__':
    env = SlimeEnvironment()
    c = 0
    while True:
        c += 1
        update_usr()
        obs, r, done = env.step(usr1, display=True)
        if r > 0: print(r)
        if done:
            print(r)
            time.sleep(0.5)
            env.reset()
        




        
