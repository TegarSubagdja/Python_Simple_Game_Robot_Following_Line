import cv2
import numpy as np
import math
from cv2 import aruco

# Parameter kamera (disesuaikan dengan kalibrasi kamera)
camera_matrix = np.array([[1000, 0, 320],
                          [0, 1000, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4,1))

# Ukuran marker dalam meter
marker_size = 0.05  # 5 cm

# Path yang akan diikuti robot (dalam koordinat dunia)
path_points = [(0.2, 0.2), (0.4, 0.3), (0.6, 0.5)]  # Contoh titik dalam meter

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def detect_aruco_and_track():
    cap = cv2.VideoCapture(0)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i in range(len(ids)):
                marker_corners = corners[i]
                objPoints = np.array([[-marker_size/2, marker_size/2, 0],
                                      [marker_size/2, marker_size/2, 0],
                                      [marker_size/2, -marker_size/2, 0],
                                      [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(objPoints, marker_corners, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if success:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_size)
                    
                    # Posisi ArUco dalam koordinat dunia
                    robot_x, robot_y, _ = tvec.flatten()
                    
                    # Cari titik path terdekat
                    closest_point = min(path_points, key=lambda p: calculate_distance((robot_x, robot_y), p))
                    
                    # Hitung jarak dan sudut
                    distance = calculate_distance((robot_x, robot_y), closest_point)
                    angle_to_path = calculate_angle((robot_x, robot_y), closest_point)
                    
                    # Konversi orientasi ArUco ke euler angles
                    rmat, _ = cv2.Rodrigues(rvec)
                    euler_angles = cv2.RQDecomp3x3(rmat)[0]
                    robot_angle = euler_angles[2]  # Rotasi yaw
                    
                    # Sudut yang perlu disesuaikan oleh robot
                    angle_correction = angle_to_path - robot_angle
                    
                    # Tampilkan info di layar
                    info_text = f"Dist: {distance:.2f}m | Angle: {angle_correction:.1f}°"
                    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('ArUco Path Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_and_track()
