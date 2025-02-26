import cv2
import numpy as np
from cv2 import aruco
import pygame
import math
import threading

# Initialize webcam and get its dimensions first
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam.")
    exit()

# Get webcam size
FRAME_WIDTH, FRAME_HEIGHT = frame.shape[1], frame.shape[0]
print(f"Webcam resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")

# Initialize pygame with matching dimensions
pygame.init()
screen = pygame.display.set_mode((FRAME_WIDTH, FRAME_HEIGHT))
pygame.display.set_caption("ArUco Path Tracking")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Variables
path_points = []
robot_pos = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
robot_angle = 0
current_target_index = 0
marker_detected = False

# Function to calculate angle between two points
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return math.degrees(math.atan2(delta_y, delta_x))

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# Thread function for ArUco detection
def aruco_thread():
    global robot_pos, robot_angle, marker_detected
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    # Camera calibration parameters
    camera_matrix = np.array([[1000, 0, FRAME_WIDTH // 2], 
                              [0, 1000, FRAME_HEIGHT // 2], 
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    marker_size = 0.05  # Size in meters
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally to handle mirror effect
        frame_display = cv2.flip(frame, 1)  # Only for display, not for detection
            
        # Convert to grayscale for detection (use original unflipped frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        marker_detected = False
        if ids is not None:
            # Draw detected markers on the display frame
            corners_display = []
            for corner in corners:
                # Mirror the corners for display
                mirrored_corner = corner.copy()
                mirrored_corner[0, :, 0] = FRAME_WIDTH - mirrored_corner[0, :, 0]
                corners_display.append(mirrored_corner)
            
            aruco.drawDetectedMarkers(frame_display, corners_display, ids)
            
            for i in range(len(ids)):
                # Get marker corners from original frame
                marker_corners = corners[i][0]
                
                # Calculate marker position (center)
                center_x = int(np.mean([c[0] for c in marker_corners]))
                center_y = int(np.mean([c[1] for c in marker_corners]))
                
                # Mirror the x-coordinate for pygame (to handle mirror effect)
                mirrored_center_x = FRAME_WIDTH - center_x
                
                # Update robot position with mirrored coordinates
                robot_pos[0] = mirrored_center_x
                robot_pos[1] = center_y
                
                # Calculate marker orientation
                # Use the first two corners to determine direction
                front_mid_x = (marker_corners[0][0] + marker_corners[1][0]) / 2
                front_mid_y = (marker_corners[0][1] + marker_corners[1][1]) / 2
                
                # Calculate angle from center to front midpoint
                dx = front_mid_x - center_x
                dy = front_mid_y - center_y
                
                # Mirror the angle calculation for correct orientation
                angle = math.degrees(math.atan2(dy, -dx))  # Negate dx to mirror the angle
                robot_angle = angle
                
                marker_detected = True
                
                # Draw orientation line on the display frame (with mirrored coordinates)
                mirrored_front_x = FRAME_WIDTH - front_mid_x
                cv2.line(frame_display, 
                        (int(mirrored_center_x), int(center_y)), 
                        (int(mirrored_center_x + 30 * math.cos(math.radians(angle))), 
                         int(center_y + 30 * math.sin(math.radians(angle)))), 
                        (0, 255, 0), 2)
        
        # Show camera feed with detections
        cv2.imshow('ArUco Detection', frame_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Start ArUco detection in a separate thread
aruco_thread = threading.Thread(target=aruco_thread)
aruco_thread.daemon = True
aruco_thread.start()

# Main loop
clock = pygame.time.Clock()
running = True
font = pygame.font.Font(None, 24)

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                path_points.append(event.pos)
                print(f"Added path point: {event.pos}")
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear path with 'c' key
                path_points = []
                current_target_index = 0
                print("Path cleared")
    
    # Clear screen
    screen.fill(WHITE)
    
    # Calculate information for current target point (if exists)
    target_angle = 0
    angle_error = 0
    distance = 0
    
    if path_points and marker_detected:
        # Find current target point
        current_target = path_points[min(current_target_index, len(path_points) - 1)]
        
        # Calculate target angle (angle robot needs to be at to point to target)
        target_angle = calculate_angle(robot_pos, current_target)
        
        # Calculate angle error (difference between robot angle and target angle)
        # Normalize to -180 to 180 range
        angle_error = (target_angle - robot_angle) % 360
        if angle_error > 180:
            angle_error -= 360
            
        # Calculate distance to target
        distance = calculate_distance(robot_pos, current_target)
        
        # Draw line to current target
        pygame.draw.line(screen, YELLOW, robot_pos, current_target, 2)
        
        # Highlight current target
        pygame.draw.circle(screen, PURPLE, current_target, 8)
        
        # Draw all target angles for visualization
        target_arrow_length = 40
        target_arrow_x = robot_pos[0] + target_arrow_length * math.cos(math.radians(target_angle))
        target_arrow_y = robot_pos[1] + target_arrow_length * math.sin(math.radians(target_angle))
        pygame.draw.line(screen, GREEN, robot_pos, (target_arrow_x, target_arrow_y), 2)
    
    # Draw path
    if len(path_points) > 1:
        pygame.draw.lines(screen, BLUE, False, path_points, 3)
    
    # Draw all path points
    for i, point in enumerate(path_points):
        color = PURPLE if i == current_target_index and current_target_index < len(path_points) else GREEN
        pygame.draw.circle(screen, color, point, 6)
    
    # Draw robot
    if marker_detected:
        # Draw robot body
        pygame.draw.circle(screen, RED, (int(robot_pos[0]), int(robot_pos[1])), 15)
        
        # Draw robot direction indicator
        arrow_length = 25
        arrow_x = robot_pos[0] + arrow_length * math.cos(math.radians(robot_angle))
        arrow_y = robot_pos[1] + arrow_length * math.sin(math.radians(robot_angle))
        pygame.draw.line(screen, BLACK, (int(robot_pos[0]), int(robot_pos[1])), (int(arrow_x), int(arrow_y)), 3)
        
        # Draw small circle at the arrow tip
        pygame.draw.circle(screen, BLACK, (int(arrow_x), int(arrow_y)), 5)
        
        # Display information about marker detection and robot state
        status_text = font.render("Marker Detected", True, (0, 100, 0))
        screen.blit(status_text, (10, 10))
        
        # Display position information
        pos_text = font.render(f"Position: ({int(robot_pos[0])}, {int(robot_pos[1])})", True, BLACK)
        screen.blit(pos_text, (10, 40))
        
        # Display angle information
        angle_text = font.render(f"Robot Angle: {robot_angle:.1f}°", True, BLACK)
        screen.blit(angle_text, (10, 70))
        
        # Display target information if there are path points
        if path_points:
            # Display target angle
            target_angle_text = font.render(f"Target Angle: {target_angle:.1f}°", True, GREEN)
            screen.blit(target_angle_text, (10, 100))
            
            # Display angle error
            error_color = GREEN if abs(angle_error) < 10 else YELLOW if abs(angle_error) < 30 else RED
            error_text = font.render(f"Angle Error: {angle_error:.1f}°", True, error_color)
            screen.blit(error_text, (10, 130))
            
            # Display distance to target
            dist_text = font.render(f"Distance to Target: {distance:.1f} pixels", True, BLUE)
            screen.blit(dist_text, (10, 160))
            
            # Display whether robot is pointing to target
            aligned = abs(angle_error) < 10  # Consider aligned if error is less than 10 degrees
            aligned_text = font.render(
                f"Status: {'Aligned to target' if aligned else 'Not aligned'}",
                True, 
                GREEN if aligned else RED
            )
            screen.blit(aligned_text, (10, 190))
    else:
        # Draw inactive robot
        pygame.draw.circle(screen, (200, 200, 200), (int(robot_pos[0]), int(robot_pos[1])), 15)
        no_marker_text = font.render("No Marker Detected", True, (200, 0, 0))
        screen.blit(no_marker_text, (10, 10))
    
    # Display instructions
    instructions = [
        "Left-click: Add waypoint",
        "C key: Clear path",
        "Q key in camera window: Quit"
    ]
    for i, instruction in enumerate(instructions):
        inst_text = font.render(instruction, True, (100, 100, 100))
        screen.blit(inst_text, (FRAME_WIDTH - 200, 10 + i * 25))
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(30)

# Cleanup
pygame.quit()