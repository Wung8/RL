import numpy as np
import cv2, math, time, random
import keyboard as k
import torch
import time

ACTION_SPACE = 32
HEIGHT, WIDTH = 500, 500
buffer = 0.05

class DiepioEnvironment:
    def __init__(self):
        self.stack = [np.zeros((80,80)) for i in range(4)]
        
        self.framerate = 20
        self.skip_frames = 3
        self.height = HEIGHT
        self.width = WIDTH
        self.center = HEIGHT/2, WIDTH/2
        
        self.turnspeed = 25

        self.tank = Tank()
        self.bullets = []
        self.hitrange = 12
        self.bullet_damage = 2
        self.bullet_spread = 10

        self.shapes = []
        self.max_shapes = 15
        self.shapes_spawn_chance = 0.1
        self.shapes_type_chance = [16, 4, 1]
        x = self.shapes_type_chance
        self.shapes_type_chance = np.divide(x, np.sum(x))
        self.shape_types = [Square, Triangle, Pentagon]

    def reset(self):
        self.stack = [np.zeros((80,80)) for i in range(4)]
        self.tank = Tank()
        self.bullets = []
        self.shapes = []
        for i in range(self.max_shapes):
            self.spawn_shape()
        return self.getInputs()

    def getInputs(self):
        frame = self.display(resolution=((80,80)), display=False)
        del self.stack[0]
        self.stack.append(frame)
        #return np.array(self.stack, dtype=np.float32)/255
        return np.array([frame], dtype=np.float32)/255

    def spawn_shape(self):
        shape = np.random.choice(self.shape_types, p=self.shapes_type_chance)()
        self.shapes.append(shape)

    def step(self, action, display=False, is_skip=False, display_skip=True):

        reward = 0
        if is_skip == False:
            #self.tank.angle = (self.tank.angle+(action-1.5)*self.turnspeed)%360
            self.tank.angle = action * 360 / ACTION_SPACE
            for i in range(self.skip_frames):
                result = self.step(action, display=display&display_skip, is_skip=True)
                reward += result

        if len(self.shapes) < self.max_shapes and random.random() < self.shapes_spawn_chance:
            self.spawn_shape()

        for i,bullet in list(enumerate(self.bullets))[::-1]:
            bullet.step()
            if bullet.health == 0:
                del self.bullets[i]
                continue
            for j,shape in list(enumerate(self.shapes))[::-1]:
                if math.dist(bullet.center,shape.center) < (bullet.s+shape.s)*0.9:
                    del self.bullets[i]
                    shape.health -= self.bullet_damage
                    if shape.health <= 0:
                        reward += shape.score
                        del self.shapes[j]
                    break

        self.tank.step()
        if self.tank.reload == self.tank.max_reload:
            bullet = Bullet(self.tank.angle + self.bullet_spread * (random.random()-0.5))
            self.bullets.append(bullet)

        if display: self.display()

        if is_skip: return reward

        return self.getInputs(), reward, 0

    def display(self, resolution=(WIDTH,HEIGHT), display=True):
        img = np.array([[[204,204,204]]], dtype=np.uint8)
        img = img.repeat(self.height,axis=0).repeat(self.width,axis=1)

        for obj in self.shapes + self.bullets:
            img = obj.display(img)

        img = self.tank.display(img)

        if display:
            #img = cv2.resize(img, (600,400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('img', img)
            cv2.waitKey(math.ceil(1000/self.framerate))
        else:
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        return img

    def get_expert_action(self):
        # pentagon first
        # triangle second
        # section with most squares third
        pentagons = []
        triangles = []
        square_counts = [0 for i in range(ACTION_SPACE)]

        for shape in self.shapes:
            angle = get_angle(*shape.center)
            section = round(angle * ACTION_SPACE / 360) % ACTION_SPACE
            if shape.name == "pentagon":
                pentagons.append((math.dist(self.center, shape.center), section))
            elif shape.name == "triangle":
                triangles.append((math.dist(self.center, shape.center), section))
            elif shape.name == "square":
                if section == 32: section = 0
                square_counts[section] += 1

        if pentagons:
            pentagons.sort()
            return pentagons[0][1]
        elif triangles:
            triangles.sort()
            return triangles[0][1]
        else:
            return np.argmax(square_counts)
            

def get_angle(x, y):
    return math.atan2(x - WIDTH//2, y - HEIGHT//2) / math.pi * 180  

def rotate(points, angle):
    rotation_matrix = cv2.getRotationMatrix2D((0,0), angle, 1)
    return np.dot(points, rotation_matrix[:, :2].T)

def generate_center():
    return (random.random()*WIDTH*(1-2*buffer)+WIDTH*buffer,
            random.random()*HEIGHT*(1-2*buffer)+HEIGHT*buffer)

def generate_points(n):
    points = []
    for i in range(n):
        points.append([math.cos(i*2*math.pi/n), math.sin(i*2*math.pi/n)])
    return points

[1,3,10]
[1,2.5,13]

class Square():
    def __init__(self):
        self.name = "square"
        self.score = 1
        self.health = 1
        self.center = generate_center()
        self.s = 13
        self.angle = random.random()*360

        points = np.array(generate_points(4),dtype=np.float32) * self.s
        self.points = rotate(points, self.angle) + self.center
        self.points = np.int32([np.round(self.points)])

    def display(self, img):
        img = cv2.fillPoly(img, self.points, color=(110,233,252))
        img = cv2.polylines(img, self.points, True, (82,175,189), 2)
        return img

class Triangle():
    def __init__(self):
        self.name = "triangle"
        self.score = 2.5
        self.health = 3
        self.center = generate_center()
        self.s = 15
        self.angle = random.random()*360

        points = np.array(generate_points(3),dtype=np.float32) * self.s
        self.points = rotate(points, self.angle) + self.center
        self.points = np.int32([np.round(self.points)])

    def display(self, img):
        img = cv2.fillPoly(img, self.points, color=(120,120,247))
        img = cv2.polylines(img, self.points, True, (94,94,195), 2)
        return img

class Pentagon():
    def __init__(self):
        self.name = "pentagon"
        self.score = 13
        self.health = 10
        self.center = generate_center()
        self.s = 18
        self.angle = random.random()*360

        points = np.array(generate_points(5),dtype=np.float32) * self.s
        self.points = rotate(points, self.angle) + self.center
        self.points = np.int32([np.round(self.points)])

    def display(self, img):
        img = cv2.fillPoly(img, np.int32([np.round(self.points)]), color=(250,139,125))
        img = cv2.polylines(img, self.points, True, (188,110,100), 2)
        return img

class Tank():
    def __init__(self):
        self.center = (WIDTH//2, HEIGHT//2)
        self.s1 = 15
        self.s2 = 15
        self.angle = 0
        self.max_reload = 7
        
        self.reload = self.max_reload

    def step(self):
        self.reload -= 1
        if self.reload == 0:
            self.reload = self.max_reload

    def display(self, img):
        points = np.array([[-.5,2],[.5,2],[.5,0],[-.5,0]], dtype=np.float32) * self.s2
        points = rotate(points, self.angle) + self.center
        points = np.int32([np.round(points)])
        img = cv2.fillPoly(img, points, color=(153,153,153))
        img = cv2.polylines(img, points, True, (114,114,114), 2)
        img = cv2.circle(img, self.center, self.s1, (224,177,50), -1)
        img = cv2.circle(img, self.center, self.s1, (167,132,36), 2)
        return img

class Bullet():
    def __init__(self, angle):
        self.health = 1
        self.center = (WIDTH//2, HEIGHT//2)
        self.s = 8
        self.speed = 20
        angle = 90-angle
        self.velocity = np.multiply((math.cos(angle/180*math.pi),math.sin(angle/180*math.pi)),self.speed)

    def step(self):
        self.center = np.add(self.center, self.velocity)
        x,y = self.center
        if x<0 or x>WIDTH: self.health = 0
        if y<0 or y>HEIGHT: self.health = 0

    def display(self, img):
        img = cv2.circle(img, np.int32(self.center), self.s, (224,177,50), -1)
        img = cv2.circle(img, np.int32(self.center), self.s, (167,132,36), 2)
        return img


MOUSE_COORDS = (0,0)

def get_mouse_coordinates(event, x, y, flags, param):
    global MOUSE_COORDS
    MOUSE_COORDS = (x,y)

if __name__ == '__main__':
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', get_mouse_coordinates)

    env = DiepioEnvironment()
    env.reset()

    while True:
        #angle = math.atan2(MOUSE_COORDS[0]-WIDTH//2, MOUSE_COORDS[1]-HEIGHT//2) / math.pi * 180
        #usr = round(angle / 360 * 32)
        usr = env.get_expert_action()
        env.step(usr, display=True)









