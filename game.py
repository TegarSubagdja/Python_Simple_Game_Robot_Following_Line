import pygame
import math

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Path Game")

# Variables
path_points = []
robot_pos = [WIDTH // 2, HEIGHT // 2]
robot_angle = 0
moving = False
target_index = 0
holding_robot = False  # Status apakah sedang memegang robot dengan klik kanan

# Function to calculate angle
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return math.degrees(math.atan2(delta_y, delta_x))

# Reset function
def reset_game():
    global path_points, robot_pos, robot_angle, moving, target_index
    path_points = []
    robot_pos = [WIDTH // 2, HEIGHT // 2]
    robot_angle = 0
    moving = False
    target_index = 0

# Main loop
running = True
while running:
    screen.fill(WHITE)
    
    # Draw path
    if len(path_points) > 1:
        pygame.draw.lines(screen, BLUE, False, path_points, 3)
    
    # Draw robot (circle)
    pygame.draw.circle(screen, RED, robot_pos, 10)
    
    # Draw direction arrow
    arrow_length = 15
    arrow_x = robot_pos[0] + arrow_length * math.cos(math.radians(robot_angle))
    arrow_y = robot_pos[1] + arrow_length * math.sin(math.radians(robot_angle))
    pygame.draw.line(screen, BLACK, robot_pos, (arrow_x, arrow_y), 3)
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click to draw path
                path_points.append(event.pos)
            elif event.button == 3:  # Right click to move robot
                # Jika klik dekat robot, aktifkan mode geser (drag)
                if math.hypot(robot_pos[0] - event.pos[0], robot_pos[1] - event.pos[1]) < 15:
                    holding_robot = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:  # Lepaskan hold
                holding_robot = False
        elif event.type == pygame.MOUSEMOTION:
            if holding_robot:  # Geser robot ketika klik kanan ditahan
                robot_pos = list(event.pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:  # Tekan SPACE untuk mulai bergerak
                if path_points:
                    moving = True
                    target_index = 0  # Reset ke titik pertama
            elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                reset_game()  # Ctrl + R untuk reset semua
    
    # Move robot along the path
    if moving and target_index < len(path_points):
        target = path_points[target_index]
        robot_angle = calculate_angle(robot_pos, target)
        dist = math.hypot(target[0] - robot_pos[0], target[1] - robot_pos[1])
        
        if dist > 3:
            robot_pos[0] += 2 * math.cos(math.radians(robot_angle))
            robot_pos[1] += 2 * math.sin(math.radians(robot_angle))
        else:
            target_index += 1  # Pindah ke titik berikutnya
            if target_index >= len(path_points):
                moving = False  # Hentikan pergerakan di titik terakhir
    
    pygame.display.flip()
    pygame.time.delay(30)

pygame.quit()