import pygame
import numpy as np
import keyboard as k


SCALE = 1.5
DT = 0.1
G = 20
COLORS = {
    'bg':(140,190,170),
    'ground':(70,100,90),
    'player':(20,20,20),
    'blue':(50,50,150),
    'green':(50,150,50)
}

class Rectangle:
    def __init__(self, pos, dim):
        self.pos = pos
        self.dim = dim

class PlatformerEnvironment:
    def __init__(self):
        self.screen_size = (800, 400)
        self.camera_speed = 0.2
        self.display_prev = False
        # everything is drawn onto surface and scaled up onto the screen
        self.surface = pygame.Surface(self.screen_size, pygame.SRCALPHA) 
        self.screen = pygame.display.set_mode(np.multiply(self.screen_size, SCALE), flags=pygame.HIDDEN)
        self.clock = pygame.time.Clock()
        self.max_tick = 120 # supports cycles of 1,2,3,4,5,6,8,9,...
        self.tick = 0

        self.player = Player(self)

        self.death_zone = (100000, -200)
        self.platforms = [
            Platform(self, (850,160), (1200,20)),
            Platform(self, (250,250), (20,200)),
            Platform(self, (600,180), (80,40)),
            Platform(self, (1000,180), (280,40)),
            Platform(self, (950,220), (80,60)),
            Platform(self, (1150,200), (80,60)),
            Platform(self, (1350,210), (80,90)),
            Platform(self, (1400,180), (60,40)),
            Platform(self, (1550,240), (80,20)),
            Platform(self, (1700,220), (60,20)),
            Platform(self, (1800,260), (40,20)),
            Platform(self, (2100,160), (300,20)),
        ]

        self.reset()

    def reset(self):
        self.tick = 0
        self.player.reset()
        self.camera_center = np.subtract(self.player.pos, np.rint(np.multiply(self.screen_size, 0.5)))

        self.player_max_x = self.player.pos[0]
        self.time_since_last_progress = 0
        return self.display(display=False)

    # take input and process frame
    def step(self, actions, display=False):
        if self.display_prev != display:
            self.display_prev = display
            self.screen = pygame.display.set_mode(np.multiply(self.screen_size, SCALE), flags=[pygame.HIDDEN, pygame.SHOWN][display])
        self.tick += 1
        if self.tick == self.max_tick: self.tick = 0
        self.player.step(actions)
        done = 0
        reward = max(self.player.pos[0] - self.player_max_x, 0) / 100
        if self.player.pos[0] > self.player_max_x:
            self.player_max_x = self.player.pos[0]
            self.time_since_last_progress = 0
        self.time_since_last_progress += 1
        if abs(self.player.pos[0]) > self.death_zone[0] or self.player.pos[1] < self.death_zone[1]:
            done = 1
            reward = -0.5
        if self.time_since_last_progress >= 250:
            done = 1
            reward = -0.5
        if self.player.pos[0] >= 2100:
            done = 1
            reward = 0.5
        
        if display: self.display()

        self.camera_center = self.camera_speed * np.subtract(self.player.pos, np.rint(np.multiply(self.screen_size, 0.5))) \
                             + (1-self.camera_speed) * self.camera_center
        obs = self.display(display=False)

        return obs, reward, done
        
    def display(self, display=True):
        pygame.event.pump()
        self.surface.fill(COLORS['bg'])

        for platform in self.platforms:
            platform.display(self.camera_center)
        self.player.display(self.camera_center)

        self.surface = pygame.transform.flip(self.surface, False, True)

        if display==False:
            obs = pygame.transform.scale(self.surface, (80,80))
            obs = np.transpose(pygame.surfarray.array3d(obs), (2,0,1))
            return obs
            
        pygame.transform.scale(self.surface, (np.multiply(self.screen_size, SCALE)), self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    # //////////////////////////////////////GAME FUNCS//////////////////////////////////////

    def move(self, pos, upd, scale=1):
        return np.rint(np.add(pos, np.multiply(upd,scale)))

    def update_position(self, player):
        player.pos = self.move(player.pos, (0,player.vel[1]), scale=DT)
        update_vec = (0,[-1,1][player.vel[1] < 0])
        # move player pos down one pixel to check for ground
        move_up = False
        if player.vel[1] < 0:
            player.pos = self.move(player.pos, (0,-1))
            move_up = True # if no platforms touched, move the player pos back up a pixel
        for platform in self.platforms:
            while self.check_rectangle_collision(player.get_rect(), platform.get_rect()):
                move_up = False
                player.pos = self.move(player.pos, update_vec)
                player.vel = player.vel[0], 0
                if update_vec[1] > 0:
                    player.jump_flag = True
        if move_up: player.pos = self.move(player.pos, (0,1))

        player.pos = self.move(player.pos, (player.vel[0],0), scale=DT)
        update_vec = ([-1,1][player.vel[0] < 0],0)
        for platform in self.platforms:
            while self.check_rectangle_collision(player.get_rect(), platform.get_rect()):
                player.pos = self.move(player.pos, update_vec)
                player.vel = 0, player.vel[1]    
    
    def check_rectangle_collision(self, rect1, rect2):
        x1, y1, w1, h1 = *rect1.pos, *rect1.dim
        x2, y2, w2, h2 = *rect2.pos, *rect2.dim
        left1, right1, top1, bottom1 = x1 - w1/2, x1 + w1/2, y1 - h1/2, y1 + h1/2
        left2, right2, top2, bottom2 = x2 - w2/2, x2 + w2/2, y2 - h2/2, y2 + h2/2

        return not (right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1)


class Platform:
    # inputs: parent, center, (w,h)
    def __init__(self, parent, pos, dims):
        self.parent = parent
        self.pos = pos
        self.dims = dims

    def display(self, camera_center):
        rect = pygame.Rect(self.pos[0]-self.dims[0]/2-camera_center[0], self.pos[1]-self.dims[1]/2-camera_center[1], *self.dims)
        pygame.draw.rect(self.parent.surface, COLORS['ground'], rect)

    def get_rect(self):
        return Rectangle(self.pos, self.dims)


class Player:
    def __init__(self, parent):
        self.parent = parent
        self.pos = 0, 0
        self.dims = 20, 30
        self.vel = 0, 0

        self.walk_speed = 15
        self.jump_power = 50
        self.friction = 0.6 # what x velocity is multiplied by each frame

    def reset(self):
        self.pos = 400, 200
        self.vel = 0, 0
        
        self.jump_flag = False # flag for if jump is available
        self.jumping_flag = False # flag for whether jump button is being held to extend jump height

    def step(self, actions):
        self.vel = (np.multiply(self.vel[0], self.friction),
                    self.vel[1])
        actions[0] -= 1
        self.vel = np.add(self.vel, (actions[0]*self.walk_speed, 0))

        if self.vel[1] < 0:
            self.jump_flag = False
            self.jumping_flag = False

        # if jump pressed and not dashing
        if actions[1] == 1:
            if self.jump_flag:
                self.jump_flag = False
                self.vel = (self.vel[0],self.jump_power)
            self.jumping_flag = True
        elif actions[1] == 0:
            # if jump button let go, remove all vertical velocity for snappiness
            if self.jumping_flag:
                self.vel = (self.vel[0], 0)
            self.jumping_flag = False

        if actions[0] != 0:
            self.facing = actions[0]
            
        self.vel = np.add(self.vel, (0,-G*DT))
        # gravity has stronger effect when falling down
        if self.vel[1] < 0: self.vel = np.add(self.vel, (0,-G*DT*0.5))

        self.actions = actions
        self.parent.update_position(self)
        
    def display(self, camera_center):
        rect = pygame.Rect(self.pos[0]-self.dims[0]/2-camera_center[0], self.pos[1]-self.dims[1]/2-camera_center[1], *self.dims)
        pygame.draw.rect(self.parent.surface, COLORS['player'], rect)

    def get_rect(self):
        return Rectangle(self.pos, self.dims)


z_hold = False
x_hold = False
c_hold = False
def get_user_actions():
    global z_hold, x_hold, c_hold
    # left:0   _:1   right:2
    # _:0   hold:1   jump:2

    actions = [1,0]
    keys = pygame.key.get_pressed()
    
    if keys[pygame.K_LEFT]:
        actions[0] += -1
    if keys[pygame.K_RIGHT]:
        actions[0] += 1
        
    if keys[pygame.K_z]:
        if not z_hold:
            actions[1] = 1
        z_hold = True
        actions[1] += 1
    else: z_hold = False

    return actions


        
if __name__ == '__main__':
    pygame.init()
    env = PlatformerEnvironment()
    c = 0
    total_r = 0
    while True:
        c += 1
        actions = get_user_actions()
        obs, rew, done = env.step(actions, display=True)
        total_r += rew
        if done:
            env.reset()
            print(c, total_r)
            c, total_r = 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

