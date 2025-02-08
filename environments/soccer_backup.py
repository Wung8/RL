import pymunk
import pygame
from pymunk import Vec2d
import math


# Set up pymunk space
space = pymunk.Space()
space.gravity = (0, 0)  # No gravity

BALL_CATEGORY = 1
PLAYER_CATEGORY = 2
WALL_CATEGORY = 4

def custom_damping(body, gravity, damping, dt):
    pymunk.Body.update_velocity(body, gravity, body.custom_damping, dt)

# Add walls
def create_wall(start, end):
    wall_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    wall_shape = pymunk.Segment(wall_body, start, end, 5)
    wall_shape.friction = 0
    wall_shape.elasticity = 0.9
    wall_shape.filter = pymunk.ShapeFilter(categories=WALL_CATEGORY,
                                           mask=BALL_CATEGORY)
    space.add(wall_body, wall_shape)

def kick_ball(player_body, ball_body, kick_strength=75, kick_range=30):
    """
    Simulate a kick on the ball with fixed power, only if the ball is within a certain distance.

    Args:
        player_body (pymunk.Body): The player's body.
        ball_body (pymunk.Body): The ball's body.
        kick_strength (float): The fixed strength of the kick.
        kick_distance (float): The maximum distance at which the kick can occur.
    """
    # Calculate the distance between the player and the ball
    distance = (ball_body.position - player_body.position).length

    # Check if the ball is within the kickable distance
    if distance <= kick_range:
        # Calculate the direction of the kick
        direction = (ball_body.position - player_body.position).normalized()

        # Apply an impulse to the ball in the direction
        impulse = direction * kick_strength
        ball_body.apply_impulse_at_world_point(impulse, ball_body.position)

# Create boundary walls
walls = [
    ((46, 96), (854, 96)),
    ((46, 96), (46, 504)),
    ((854, 504), (46, 504)),
    ((854, 504), (854, 96))
]
for start, end in walls:
    create_wall(start, end)

# Create controllable circle
controllable_body = pymunk.Body(1, math.inf)
controllable_body.position = 400, 300
controllable_body.custom_damping = 0.95
controllable_body.velocity_func = custom_damping
controllable_shape = pymunk.Circle(controllable_body, 15)
controllable_shape.friction = 0  # No friction
controllable_shape.elasticity = 0.5
controllable_shape.name = "blue"
controllable_shape.filter = pymunk.ShapeFilter(categories=PLAYER_CATEGORY,
                                               mask=PLAYER_CATEGORY | BALL_CATEGORY)
space.add(controllable_body, controllable_shape)

# Create another circle for interaction
other_body = pymunk.Body(0.5, math.inf)
other_body.position = 500, 300
other_body.custom_damping = 0.985
other_body.velocity_func = custom_damping
other_shape = pymunk.Circle(other_body, 11)
other_shape.friction = 0  # No friction
other_shape.elasticity = 1
other_shape.name = "ball"
other_shape.filter = pymunk.ShapeFilter(categories=BALL_CATEGORY,
                                        mask=PLAYER_CATEGORY | WALL_CATEGORY)
space.add(other_body, other_shape)

# Control variables
force = 450  # Force to apply for movement

# Main loop

# Rectangle coordinates
top_left = (50, 100)
bottom_right = (850, 500)

# Calculate the width and height
width = bottom_right[0] - top_left[0]
height = bottom_right[1] - top_left[1]

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((900, 600))
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        controllable_body.apply_force_at_local_point(Vec2d(-force, 0))
    if keys[pygame.K_RIGHT]:
        controllable_body.apply_force_at_local_point(Vec2d(force, 0))
    if keys[pygame.K_UP]:
        controllable_body.apply_force_at_local_point(Vec2d(0, -force))
    if keys[pygame.K_DOWN]:
        controllable_body.apply_force_at_local_point(Vec2d(0, force))
    if keys[pygame.K_x]:
        kick_ball(controllable_body, other_body)

    # Clear screen
    screen.fill((93,127,102))
    pygame.draw.rect(screen, (120,152,128), (top_left[0], top_left[1], width, height), 2)

    # Draw shapes
    for shape in space.shapes:
        if isinstance(shape, pymunk.Circle):
            if shape.name == "blue":
                color = (96,142,232)
            elif shape.name == "red":
                color = (207,52,52)
            else:
                color = (255,255,255)
            pos = shape.body.position
            pygame.draw.circle(screen, color, (int(pos.x), int(pos.y)), int(shape.radius), 0)

    # Update physics
    space.step(1 / 60.0)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
