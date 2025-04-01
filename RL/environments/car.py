import pygame
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Clock
clock = pygame.time.Clock()

# Car properties
car_width, car_height = 20, 12
acceleration = 0.15
friction = 0.05
max_velocity = 5
turn_speed = 3

# Irregular track points (inner and outer walls)
inner_wall_points = [
    (153, 496), (91, 439), (90, 342), (117, 286), (116, 234),
    (77, 200), (80, 101), (116, 65), (367, 59), (442, 97),
    (450, 157), (327, 260), (465, 233), (537, 79),
    (716, 75), (737, 140), (717, 184), (641, 227), (616, 283),
    (641, 361), (730, 327), (756, 383), (749, 523), (639, 556),
    (528, 525), (434, 528), (374, 552), (207, 543), (153, 496)]
outer_wall_points = [
    (158, 404), (171, 303), (161, 179), (178, 125),
    (337, 110), (367, 126), (238, 252), (227, 304), (251, 343),
    (503, 291), (572, 145), (670, 133), (615, 167), (561, 283),
    (576, 392), (641, 433), (691, 408), (691, 458),
    (602, 491), (432, 456), (347, 485), (241, 482), (158, 404)]
checkpoints = [(71, 345), (188, 347), (89, 259), (188, 258),
               (69, 146), (186, 143), (171, 47), (206, 144),
               (323, 38), (322, 132), (335, 135), (462, 136),
               (292, 174), (385, 226), (208, 312), (353, 250),
               (347, 250), (331, 338), (454, 211), (506, 312),
               (528, 83), (583, 173), (640, 148), (730, 66),
               (628, 147), (702, 202), (553, 267), (652, 271),
               (649, 343), (592, 425), (741, 301), (676, 432),
               (669, 454), (765, 541), (664, 572), (585, 459),
               (443, 436), (457, 562), (307, 459), (262, 568),
               (180, 394), (69, 466)]

# Convert points into line segments
inner_wall_lines = [(inner_wall_points[i], inner_wall_points[i + 1]) for i in range(len(inner_wall_points) - 1)]
outer_wall_lines = [(outer_wall_points[i], outer_wall_points[i + 1]) for i in range(len(outer_wall_points) - 1)]

# Car class
class CarEnv:
    def __init__(self):
        self.spawn = (107, 400)
        self.x, self.y = self.spawn
        self.angle = 90
        self.x_velocity = 0
        self.y_velocity = 0
        self.rect = pygame.Rect(self.x, self.y, car_width, car_height)
        self.checkpoint = 0

    def draw(self, surface):
        # Rotate the car
        rotated_car = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        pygame.draw.rect(rotated_car, BLUE, (0, 0, car_width, car_height))
        rotated_car = pygame.transform.rotate(rotated_car, self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

        # Draw heading line
        end_x = self.x + math.cos(math.radians(self.angle)) * 20
        end_y = self.y - math.sin(math.radians(self.angle)) * 20
        pygame.draw.line(surface, RED, (self.x, self.y), (end_x, end_y), 2)

    def reset(self):
        self.x, self.y = self.spawn
        self.angle = 90
        self.x_velocity = 0
        self.y_velocity = 0
        self.checkpoint = 0

    def update(self, keys):
        # Apply acceleration in the direction of the car's angle
        if keys[pygame.K_UP]:
            self.x_velocity += math.cos(math.radians(self.angle)) * acceleration
            self.y_velocity -= math.sin(math.radians(self.angle)) * acceleration
        if keys[pygame.K_DOWN]:
            self.x_velocity -= math.cos(math.radians(self.angle)) * acceleration * 0.5
            self.y_velocity += math.sin(math.radians(self.angle)) * acceleration * 0.5

        # Apply friction to reduce velocity
        self.x_velocity *= (1 - friction)
        self.y_velocity *= (1 - friction)

        # Clamp velocity to max speed
        velocity_magnitude = math.sqrt(self.x_velocity**2 + self.y_velocity**2)
        if velocity_magnitude > max_velocity:
            scaling_factor = max_velocity / velocity_magnitude
            self.x_velocity *= scaling_factor
            self.y_velocity *= scaling_factor

        # Update position
        self.x += self.x_velocity
        self.y += self.y_velocity

        # Turn the car (only when moving)
        if keys[pygame.K_LEFT]:
            self.angle += turn_speed * (1 if velocity_magnitude > 0 else 0)
        if keys[pygame.K_RIGHT]:
            self.angle -= turn_speed * (1 if velocity_magnitude > 0 else 0)

        # Update the rect for collision detection
        self.rect = pygame.Rect(self.x - car_width // 2, self.y - car_height // 2, car_width, car_height)

    def check_collision(self, lines):
        # Check if any of the car's corners intersect with the lines
        corners = [
            (self.x - car_width // 2, self.y - car_height // 2),
            (self.x + car_width // 2, self.y - car_height // 2),
            (self.x - car_width // 2, self.y + car_height // 2),
            (self.x + car_width // 2, self.y + car_height // 2)
        ]
        sides = list(zip(corners, (corners*2)[1:]))
        for line in lines:
            for side in sides:
                if line_collision(*line, *side):
                    return True
        return False

    def cast_ray(self, direction, lines):
        """Cast a ray in the given direction and return the closest intersection."""
        ray_end_x = self.x + math.cos(math.radians(direction)) * 1000
        ray_end_y = self.y - math.sin(math.radians(direction)) * 1000
        closest_point = None
        min_distance = float('inf')

        for start, end in lines:
            intersection = line_intersection((self.x, self.y), (ray_end_x, ray_end_y), start, end)
            if intersection:
                distance = math.hypot(intersection[0] - self.x, intersection[1] - self.y)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = intersection

        return closest_point

    def draw_rays(self, surface, lines):
        """Draw rays in multiple directions."""
        for offset in [i for i in range(0,360,12)]:  # Left, center, right rays
            direction = self.angle + offset
            intersection = self.cast_ray(direction, lines)
            if intersection:
                pygame.draw.line(surface, RED, (self.x, self.y), intersection, 2)

def line_intersection(p1, p2, p3, p4):
    """Calculate the intersection point of two line segments."""
    x1, y1, x2, y2 = *p1, *p2
    x3, y3, x4, y4 = *p3, *p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Parallel or nearly parallel lines
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # Handle horizontal rays
    if abs(y1 - y2) < 1e-6:  # Ray is horizontal
        if not (min(x3, x4) <= px <= max(x3, x4)):
            return None
        else: return px, py

    # Handle vertical rays
    if abs(x1 - x2) < 1e-6:  # Ray is vertical
        if not (min(y3, y4) <= py <= max(y3, y4)):
            return None
        else: return px, py

    # Check bounds for both segments
    if not (min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)):
        return None
    if not (min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4)):
        return None

    return px, py

def line_collision(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    def orientation(px, py, qx, qy, rx, ry):
        """Find the orientation of the ordered triplet (p, q, r)."""
        val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else -1  # Clockwise or Counterclockwise

    def on_segment(px, py, qx, qy, rx, ry):
        """Check if point (r) lies on segment (p-q)."""
        return min(px, qx) <= rx <= max(px, qx) and min(py, qy) <= ry <= max(py, qy)

    # Find the four orientations needed for the general and special cases
    o1 = orientation(x1, y1, x2, y2, x3, y3)
    o2 = orientation(x1, y1, x2, y2, x4, y4)
    o3 = orientation(x3, y3, x4, y4, x1, y1)
    o4 = orientation(x3, y3, x4, y4, x2, y2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special case: Check if the segments are collinear and overlap
    if o1 == 0 and on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if o2 == 0 and on_segment(x1, y1, x2, y2, x4, y4):
        return True
    if o3 == 0 and on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if o4 == 0 and on_segment(x3, y3, x4, y4, x2, y2):
        return True

    return False

# Main game loop
def main():
    car = CarEnv()  # Start near the track
    running = True

    while running:
        screen.fill(WHITE)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Key presses
        keys = pygame.key.get_pressed()
        car.update(keys)

        # Draw racetrack walls
        for start, end in inner_wall_lines:
            pygame.draw.line(screen, BLACK, start, end, 5)
        for start, end in outer_wall_lines:
            pygame.draw.line(screen, BLACK, start, end, 5)

        pygame.draw.line(screen, GREEN, checkpoints[car.checkpoint*2], checkpoints[car.checkpoint*2+1], 3)
        car.draw_rays(screen, inner_wall_lines + outer_wall_lines)

        # Draw car
        car.draw(screen)

        # Check collisions
        if car.check_collision(inner_wall_lines) or car.check_collision(outer_wall_lines):
            car.reset()

        if car.check_collision([(checkpoints[car.checkpoint*2], checkpoints[car.checkpoint*2+1])]):
            car.checkpoint += 1
            if car.checkpoint == len(checkpoints)/2: car.checkpoint = 0

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
