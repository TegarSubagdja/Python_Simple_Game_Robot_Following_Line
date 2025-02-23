import cv2
import numpy as np
from cv2 import aruco

def detect_aruco_pose_realtime():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)
    
    # Membuat dictionary ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    # Parameter kamera (sesuaikan dengan kamera Anda)
    camera_matrix = np.array([[1000, 0, 320],
                            [0, 1000, 240],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    
    # Ukuran marker dalam meter
    marker_size = 0.05  # 5 cm
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None:
            # Gambar marker
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Estimasi pose untuk setiap marker
            for i in range(len(ids)):
                # Dapatkan 4 corner points dari marker
                marker_corners = corners[i]
                objPoints = np.array([[-marker_size/2, marker_size/2, 0],
                                    [marker_size/2, marker_size/2, 0],
                                    [marker_size/2, -marker_size/2, 0],
                                    [-marker_size/2, -marker_size/2, 0]], dtype=np.float32)
                
                # Estimasi pose menggunakan solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    objPoints,
                    marker_corners,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    # Gambar axes
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                                    rvec, tvec, marker_size)
                    
                    # Dapatkan koordinat 3D
                    x, y, z = tvec.flatten()
                    
                    # Konversi rotasi vector ke matrix rotasi dan euler angles
                    rmat = cv2.Rodrigues(rvec)[0]
                    euler_angles = cv2.RQDecomp3x3(rmat)[0]
                    
                    # Tampilkan informasi pose
                    marker_id = ids[i][0]
                    pos_text = f"ID {marker_id} Pos: x={x:.2f}, y={y:.2f}, z={z:.2f}"
                    rot_text = f"Rot: rx={euler_angles[0]:.1f}, ry={euler_angles[1]:.1f}, rz={euler_angles[2]:.1f}"
                    
                    cv2.putText(frame, pos_text, (10, 30 + 30*i), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, rot_text, (10, 50 + 30*i),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('ArUco 3D Pose Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_pose_realtime()