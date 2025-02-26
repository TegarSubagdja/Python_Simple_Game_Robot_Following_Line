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

# Variables
path_points = []
robot_pos = [FRAME_WIDTH // 2, FRAME_HEIGHT // 2]
robot_angle = 0
moving = False
target_index = 0
marker_detected = False

# Function to calculate angle between two points
def calculate_angle(p1, p2):
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return math.degrees(math.atan2(delta_y, delta_x))

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
                print("Path cleared")
    
    # Clear screen
    screen.fill(WHITE)
    
    # Draw path
    if len(path_points) > 1:
        pygame.draw.lines(screen, BLUE, False, path_points, 3)
        # Draw points along the path
        for point in path_points:
            pygame.draw.circle(screen, GREEN, point, 5)
    
    # Draw robot
    if marker_detected:
        # Draw robot body
        pygame.draw.circle(screen, RED, robot_pos, 15)
        
        # Draw direction indicator
        arrow_length = 25
        arrow_x = robot_pos[0] + arrow_length * math.cos(math.radians(robot_angle))
        arrow_y = robot_pos[1] + arrow_length * math.sin(math.radians(robot_angle))
        pygame.draw.line(screen, BLACK, robot_pos, (arrow_x, arrow_y), 3)
        
        # Draw small circle at the arrow tip to make it more visible
        pygame.draw.circle(screen, BLACK, (int(arrow_x), int(arrow_y)), 5)
        
        # Add text showing detection status
        font = pygame.font.Font(None, 36)
        text = font.render("Marker Detected", True, (0, 100, 0))
        screen.blit(text, (10, 10))
    else:
        # Draw inactive robot
        pygame.draw.circle(screen, (200, 200, 200), robot_pos, 15)
        font = pygame.font.Font(None, 36)
        text = font.render("No Marker Detected", True, (200, 0, 0))
        screen.blit(text, (10, 10))
    
    # Update display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(30)

# Cleanup
pygame.quit()