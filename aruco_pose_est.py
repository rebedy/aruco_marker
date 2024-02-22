import csv
import cv2
import numpy as np
from glob import glob


def pose_estimation(frame, aruco_type, length, camx, dist, name, marker_data_list):
    """
    This function estimates the pose of the ArUco markers in the input frame.
    :param frame: The input frame.
    :param aruco_type: The type of the ArUco markers.
    :param length: The length of the markers' sides in meters.
    :param camx: The camera matrix.
    :param dist: The distortion coefficients.
    :param name: The name of the marker.
    :param marker_data_list: The list of marker data.
    :return: The frame with the detected markers and their poses, and the list of marker data.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters,
        cameraMatrix=camx, distCoeff=dist)

    if len(corners) > 0:
        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], length, camx, dist)
            # Convert rotation vector to rotation matrix
            rot_mat, _ = cv2.Rodrigues(rvec)
            # Calculate the Euler angles from the rotation matrix
            sy = np.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
                y = np.arctan2(-rot_mat[2, 0], sy)
                z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
                y = np.arctan2(-rot_mat[2, 0], sy)
                z = 0
            dst = np.sqrt(tvec[i][0][2] ** 2 + tvec[i][0][0] ** 2 + tvec[i][0][1] ** 2)

            angle = np.degrees([x, y, z])
            cv2.putText(frame, f"Dist: {round(dst, 2)}", (10, 30), font, 1, color, 1, line)
            cv2.putText(frame, "x, y, z:", (10, 50), font, 1, color, 1, line)
            cv2.putText(frame, f"{[[round(x, 2) for x in k] for k in tvec[0]]}", (10, 70), font, 1, color, 1, line)
            cv2.putText(frame, f"X Pitch: {round(angle[0], 2)}", (10, 90), font, 1, color, 1, line)
            cv2.putText(frame, f"Y Yaw: {round(angle[1], 2)}", (10, 110), font, 1, color, 1, line)
            cv2.putText(frame, f"Z Roll: {round(angle[2], 2)}", (10, 130), font, 1, color, 1, line)

            marker_data = [name, angle[0], angle[1], angle[2]]
            marker_data_list.append(marker_data)

            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.aruco.drawAxis(frame, camx, dist, rvec, tvec, 0.05)

    return frame, marker_data_list


def csv_save(path, all_marker_data):
    """
    This function saves the marker data to a CSV file.
    :param path: The path of the CSV file.
    :param all_marker_data: The list of marker data.
    """
    with open(path, 'w+', newline='') as csvfile:
        header = ["name", "euler_x", "euler_y", "euler_z"]
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in all_marker_data:
            writer.writerows(row)


color = (0, 0, 255)
line = cv2.LINE_AA
font = cv2.FONT_HERSHEY_PLAIN

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()

size_nums = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
lengths = [3.527777778, 5.291666667, 7.055555556, 8.819444444, 10.583333333,
           12.347222222, 14.111111111, 15.875, 17.638888889, 19.402777778,
           21.166666667, 22.930555556, 24.694444444]

for idx, size_num in enumerate(size_nums):

    img_paths = sorted(sorted(glob("distorted_markers/"+str(size_num)+"/*.png")))

    # output_path = os.path.join("recorded_output/"+str(size_num))
    # os.makedirs(output_path, exist_ok=True)

    center = (size_num/2, size_num/2)
    cam_matrix = np.array([[size_num, 0, center[0]], [0, size_num, center[1]], [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1))  # No distortion

    all_marker_data = []
    for img_path in img_paths:
        name = img_path.split("/")[-1]

        img = cv2.imread(img_path)

        marker_data_list = []
        output, marker_data_list = pose_estimation(img, cv2.aruco.DICT_6X6_250, lengths[idx], cam_matrix, dist_coeffs, name, marker_data_list)
        all_marker_data.append(marker_data_list)

        # cv2.imwrite(os.path.join(output_path, img_path.split("/")[-1]), output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    csv_save('recorded_angles/'+str(size_num)+'_data.csv', all_marker_data)
