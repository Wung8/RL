import pymunk
import pygame
from pymunk import Vec2d
import math, random
import numpy as np

BALL_CATEGORY = 1
PLAYER_CATEGORY = 2
WALL_CATEGORY = 4
WALL2_CATEGORY = 8
BLUE = (94,156,243)
RED = (195,135,95)
WHITE = (255,255,255)
LINES = (120,152,128)

def custom_damping(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, body.custom_damping, dt)

def norm(vec):
    mag = np.linalg.norm(vec)
    if mag < 1e-6: return np.multiply(vec,0)
    return np.divide(vec, mag)

def process_reward(lst):
    return np.array(lst) - np.average(lst)

class SoccerEnv:

    def __init__(self):
        # Set up pymunk space
        self.reward_scaling = 10
        self.reward_decay = 1
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity

        # Create boundary walls
        walls = [
            ((46, 96), (854, 96)),
            ((46, 96), (46, 225)),
            ((46, 375), (46, 504)),
            ((854, 504), (46, 504)),
            ((854, 504), (854, 375)),
            ((854, 225), (854, 96))
        ]
        for start, end in walls:
            self.create_wall(start, end, WALL_CATEGORY)

        walls = [
            ((0,0), (900,0)),
            ((0,0), (0,600)),
            ((900,600), (900,0)),
            ((900,600), (0,600))
        ]
        for start, end in walls:
            self.create_wall(start, end, WALL2_CATEGORY)

        posts = [
            (50,225,"darkblue"),
            (50,375,"darkblue"),
            (850,225,"darkred"),
            (850,375,"darkred")
        ]
        for x,y,color in posts:
            self.create_post((x,y),color)

        # Create players
        players = [
            (200, 200),
            (200, 400),
            (700, 200),
            (700, 400)
        ]
        self.players = []
        for pos in players:
            self.players.append(Player(self, self.space, pos))

        # Create ball
        self.ball_body = pymunk.Body(0.5, math.inf)
        self.ball_body.position = 450, 300
        self.ball_body.custom_damping = 0.985
        self.ball_body.velocity_func = custom_damping
        self.ball_shape = pymunk.Circle(self.ball_body, 11)
        self.ball_shape.friction = 0  # No friction
        self.ball_shape.elasticity = 1
        self.ball_shape.name = "ball"
        self.ball_shape.filter = pymunk.ShapeFilter(categories=BALL_CATEGORY,
                                                mask=PLAYER_CATEGORY | BALL_CATEGORY | WALL_CATEGORY)
        self.space.add(self.ball_body, self.ball_shape)

        # Initialize pygame
        self.screen = pygame.display.set_mode((900, 600), flags=pygame.HIDDEN)
        self.clock = pygame.time.Clock()
        self.display_prev = False
        
        self.reset()

    # Add walls
    def create_wall(self, start, end, mask):
        wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        wall_shape = pymunk.Segment(wall_body, start, end, 5)
        wall_shape.friction = 0
        wall_shape.elasticity = 0.8
        wall_shape.filter = pymunk.ShapeFilter(categories=mask)
        self.space.add(wall_body, wall_shape)

    def create_post(self, pos, color):
        post_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        post_body.position = pos
        post_shape = pymunk.Circle(post_body, 11)
        post_shape.friction = 0
        post_shape.elasticity = 0.8
        post_shape.filter = pymunk.ShapeFilter(categories=BALL_CATEGORY)
        post_shape.name = color
        post_shape.kicking = False
        self.space.add(post_body, post_shape)

    def kick_ball(self, player_body, kick_strength=75, kick_range=30):
        """
        Simulate a kick on the ball with fixed power, only if the ball is within a certain distance.

        Args:
            player_body (pymunk.Body): The player's body.
            ball_body (pymunk.Body): The ball's body.
            kick_strength (float): The fixed strength of the kick.
            kick_distance (float): The maximum distance at which the kick can occur.
        """
        # Calculate the distance between the player and the ball
        distance = (self.ball_body.position - player_body.position).length

        # Check if the ball is within the kickable distance
        if distance <= kick_range:
            # Calculate the direction of the kick
            direction = (self.ball_body.position - player_body.position).normalized()

            # Apply an impulse to the ball in the direction
            impulse = direction * kick_strength
            self.ball_body.apply_impulse_at_world_point(impulse, self.ball_body.position)

    # change reward to discourage moving the ball out of the corner and in front of own goal
    # maybe make it so moving ball towards goal is more heavily punished when the goal is closer
    def get_rewards(self, player):
        delta_player = np.subtract(self.players[player].body.position, self.players[player].body_last_position)
        delta_ball = np.subtract(self.ball_body.position, self.last_ball_position)
        team_goal = ([50,850][player>1.5],300)
        opp_goal = ([850,50][player>1.5],300)
        return (
            np.dot(delta_player, norm(np.subtract(self.ball_body.position, self.players[player].body.position))) * 0.005 * 0.2 # player moving to ball
            + np.dot(delta_ball, norm( np.subtract(opp_goal, self.ball_body.position))
                     * math.pow( 1-(math.dist(opp_goal, self.ball_body.position) / 1500), 1 ) * 0.010 * 0.2 ) # ball moving to opponent's goal
            - np.dot(delta_ball, norm( np.subtract(team_goal, self.ball_body.position))
                     * math.pow( 1-(math.dist(team_goal, self.ball_body.position) / 1500), 1 ) * 0.010 * 0.2 ) # ball moving to team's goal 
            - 0.001 # existential penalty
        )

    def get_angle(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.atan2(dy, dx)
        return (math.cos(angle), math.sin(angle))

    # add distance inputs, ie distance from the goals, maybe distance from the players?
    def get_inputs(self, player):
        idxs = [0,1,2,3]
        if random.random() > 0.5: # swap opponents
            if player > 1.5: idxs = [1,0,2,3]
            else: idxs = [0,1,3,2]

        p = self.players[player].body
        angles = []
        dists = []
        for i in idxs:
            if i == player: continue
            angles += self.get_angle(p.position, self.players[i].body.position)
            dists += [math.dist(p.position, self.players[i].body.position) / 900]
        angles += self.get_angle(p.position, self.ball_body.position)
        for goal in [(50,225),(50,375),(850,225),(850,375)]:
            angles += self.get_angle(p.position, goal)
            dists += [math.dist(p.position, goal) / 900]
        return [
            *angles,
            *dists,
            *self.players[idxs[0]].get_inputs(),
            *self.players[idxs[1]].get_inputs(),
            *self.players[idxs[2]].get_inputs(),
            *self.players[idxs[3]].get_inputs(),
            self.ball_body.position[0]/450-1,
            self.ball_body.position[1]/300-1,
            self.ball_body.velocity[0]/300,
            self.ball_body.velocity[1]/300
        ]
        

    def reset(self):
        self.reward_scaling = max(self.reward_scaling * self.reward_decay, 1)
        self.ball_body.position = 450, 300
        self.last_ball_position = (450, 300)
        self.ball_body.velocity = 0, 0
        for player in self.players:
            player.reset()
        return [self.get_inputs(i) for i in range(4)]

    def step(self, actions, display=False):
        self.last_ball_position = self.ball_body.position
        if self.display_prev != display:
            self.screen = pygame.display.set_mode((900, 600), flags=[pygame.HIDDEN, pygame.SHOWN][display])
        for i in range(4):
            for j in range(4):
                self.players[j].step(actions[j])
            self.space.step(1 / 60)
            if display:
                self.display_prev = True
                self.display()
            else:
                self.display_prev = False

        if self.ball_body.position[0] < 50:
            return [self.get_inputs(i) for i in range(4)], np.array([-1,-1,1,1])*self.reward_scaling, True
        elif self.ball_body.position[0] > 850:
            return [self.get_inputs(i) for i in range(4)], np.array([1,1,-1,-1])*self.reward_scaling, True
        return [self.get_inputs(i) for i in range(4)], [self.get_rewards(i) for i in range(4)], False
            

    def display(self):
        pygame.event.pump()
        # Rectangle coordinates
        top_left = (50, 100)
        bottom_right = (850, 500)

        # Calculate the width and height
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]

        # Clear screen
        self.screen.fill((93,127,102))
        for i in range(21):
            offset = 65
            pygame.draw.line(self.screen, (96,130,105), (i*offset, 0), (i*offset - 600, 600), 30)
        pygame.draw.rect(self.screen, (93,127,102), (0, 0, 900, 100), 0)
        pygame.draw.rect(self.screen, (93,127,102), (0, 0, 50, 600), 0)
        pygame.draw.rect(self.screen, (93,127,102), (850, 0, 50, 600), 0)
        pygame.draw.rect(self.screen, (93,127,102), (0, 500, 900, 100), 0)
        
        pygame.draw.rect(self.screen, LINES, (top_left[0], top_left[1], width, height), 2)
        pygame.draw.line(self.screen, LINES, (450, 100), (450, 500), 2)
        pygame.draw.rect(self.screen, LINES, (50, 150, 75, 300), 2)
        pygame.draw.rect(self.screen, LINES, (775, 150, 75, 300), 2)
        pygame.draw.circle(self.screen, (120,152,128), (450, 300), 75, 2)

        pygame.draw.circle(self.screen, LINES, (50, 225), 13, 0)
        pygame.draw.circle(self.screen, LINES, (50, 375), 13, 0)
        pygame.draw.circle(self.screen, LINES, (850, 225), 13, 0)
        pygame.draw.circle(self.screen, LINES, (850, 375), 13, 0)

        # Draw shapes
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Circle):
                if shape.name != "ball":
                    pos = shape.body.position
                    radius = shape.radius + shape.kicking * 2
                    pygame.draw.circle(self.screen, [255,255,255], (int(pos.x), int(pos.y)), int(radius), 0)
                
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Circle):
                if "blue" in shape.name:
                    color = BLUE
                elif "red" in shape.name:
                    color = RED
                else:
                    color = (255,255,255)
                if "dark" in shape.name:
                    color = np.multiply(color, 0.8)
                pos = shape.body.position
                pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), int(shape.radius), 0)

        pygame.display.flip()
        self.clock.tick(60)
        
        

class Player:

    def __init__(self, parent, space, pos):
        self.parent = parent
        self.default_pos = pos
        self.body = pymunk.Body(1, math.inf)
        self.body.position = 400, 300
        self.body.custom_damping = 0.95
        self.body.velocity_func = custom_damping
        self.shape = pymunk.Circle(self.body, 15)
        self.shape.friction = 0  # No friction
        self.shape.elasticity = 0.5
        self.shape.name = ["blue","red"][self.default_pos[0] > 450]
        self.shape.filter = pymunk.ShapeFilter(categories=PLAYER_CATEGORY,
                                                       mask=PLAYER_CATEGORY | BALL_CATEGORY | WALL2_CATEGORY)
        space.add(self.body, self.shape)
        self.force = 450
        self.shape.kicking = 0

        self.reset()

    def reset(self):
        self.body.velocity = 0, 0
        self.body_last_position = self.default_pos
        self.body.position = self.default_pos
        self.shape.kicking = 0

    def step(self, actions):
        #print(actions)
        # (left/right, up/down, kick)
        self.body_last_position = self.body.position
        self.body.apply_force_at_local_point(Vec2d(self.force * (actions[0]-1), 0))
        self.body.apply_force_at_local_point(Vec2d(0, self.force * (actions[1]-1)))
        if actions[2]:
            self.shape.kicking = 1
            self.parent.kick_ball(self.body)
            self.body.mass = 1.75
        else:
            self.shape.kicking = 0
            self.body.mass = 1

    def get_inputs(self):
        return [
            self.body.position[0]/450-1,
            self.body.position[1]/300-1,
            self.body.velocity[0]/300,
            self.body.velocity[1]/300,
            self.shape.kicking
        ]
        

if __name__ == "__main__":
    # Main loop
    pygame.init()
    env = SoccerEnv()
    c = 0
    while 1:
        c += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Controls
        actions = [[1,1,0] for i in range(4)]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            actions[3][0] -= 1
        if keys[pygame.K_RIGHT]:
            actions[3][0] += 1
        if keys[pygame.K_UP]:
            actions[3][1] -= 1
        if keys[pygame.K_DOWN]:
            actions[3][1] += 1
        if keys[pygame.K_PERIOD]:
            actions[3][2] = 1

        if keys[pygame.K_j]:
            actions[2][0] -= 1
        if keys[pygame.K_l]:
            actions[2][0] += 1
        if keys[pygame.K_i]:
            actions[2][1] -= 1
        if keys[pygame.K_k]:
            actions[2][1] += 1
        if keys[pygame.K_u]:
            actions[2][2] = 1

        if keys[pygame.K_f]:
            actions[1][0] -= 1
        if keys[pygame.K_h]:
            actions[1][0] += 1
        if keys[pygame.K_t]:
            actions[1][1] -= 1
        if keys[pygame.K_g]:
            actions[1][1] += 1
        if keys[pygame.K_r]:
            actions[1][2] = 1

        if keys[pygame.K_a]:
            actions[0][0] -= 1
        if keys[pygame.K_d]:
            actions[0][0] += 1
        if keys[pygame.K_w]:
            actions[0][1] -= 1
        if keys[pygame.K_s]:
            actions[0][1] += 1
        if keys[pygame.K_q]:
            actions[0][2] = 1

        
        obs, reward, done = env.step(actions, display=True)
        if c % 10 == 0:
            print(reward)
            #print(np.subtract(env.last_ball_pos, env.ball_body.position))
            #print(env.ball_body.position, env.ball_body.velocity)
            #print(obs[0])
        if done:
            env.reset()

    pygame.quit()






