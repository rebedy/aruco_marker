import cv2
import numpy as np


def EulerAngles(R):
    # Calculate rotation angles from rotation matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class ArucoPoseEstimator:
    """
    A class for estimating the pose of ArUco markers in a video stream.
    xs"""
    def __init__(self, camera_matrix, dist_coeffs):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.params = cv2.aruco.DetectorParameters_create()
        self.camx = camera_matrix
        self.distCoeff = dist_coeffs

    def detect_marker(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        return corners, ids

    def estimate_pose(self, corners, marker_length=1.0):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, self.camx,
                                                              self.distCoeff)
        return rvecs, tvecs

    def estimate_distance(self, corners, marker_length):
        corner = corners[0][0]
        d_pixels = np.linalg.norm(corner[0] - corner[1])
        focal_length = self.camx[0, 0]
        distance_meters = (focal_length * marker_length) / d_pixels

        return distance_meters * 100  # Convert to cm

    def draw_axis(self, frame, corners, ids, rvecs, tvecs):
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(rvecs)):
            cv2.aruco.drawAxis(frame, self.camx, self.distCoeff, rvecs[i], tvecs[i], 0.23)
            rotM = cv2.Rodrigues(rvecs[i])[0]
            angles = EulerAngles(rotM) * (180 / np.pi)
            cv2.putText(frame,
                        f"Pitch:{angles[0]:.4f}, Yaw:{angles[1]:.4f}, Roll:{angles[2]:.4f}",
                        (30, 30 + i * 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 0, 255),
                        cv2.LINE_AA)

            if len(corners) > 0:
                distance_cm = self.estimate_distance(corners, 0.1)  # 0.1 meter
                cv2.putText(frame,
                            f"Distance: {distance_cm:.2f} cm",
                            (30, 60 + i * 60),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 255, 0),
                            cv2.LINE_AA)


if __name__ == "__main__":

    # ### Video Capture ###
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter('recorded/result.avi', fourcc, 10, (w, h))

    # ### Camera Matrix and Distortion Coefficients ###
    camera_matrix = np.array([[532.46305079, 0, 332.79571817],
                              [0, 533.80458011, 221.00816556],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeff = np.array([-0.39688559,
                           0.17036189,
                           0.00482907,
                           0.0006105,
                           -0.00245277], dtype=np.float32)

    # ### Aruco Pose Estimator ###
    arucoEstimator = ArucoPoseEstimator(camera_matrix, dist_coeff)

    while True:
        # read the frame from the camera
        ret, frame = cap.read()
        # detect the markers
        corners, ids = arucoEstimator.detect_marker(frame)

        if len(corners) > 0:
            rvecs, tvecs = arucoEstimator.estimate_pose(corners)
            arucoEstimator.draw_axis(frame, corners, ids, rvecs, tvecs)

        video_writer.write(frame)
        cv2.imshow("Aruco Marker Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
