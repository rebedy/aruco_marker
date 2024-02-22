# aruco_marker

This application utilizes OpenCV to generate and detect ArUco markers of various sizes, and to estimate their 3D angles relative to the camera.



## Overview

The program is designed to perform the following tasks:

1. It generates ArUco markers, with the option to select the marker type.
2. It detects ArUco markers in real-time and estimates their 3D pose, including both orientation and position.
3. It calculates and displays the pitch, yaw, and roll angles for the detected markers.
4. It estimates and presents the distance from the camera to each marker.
5. Optionally, it preprocesses (either blurs or distorts) the generated ArUco marker if not in real-time, then returns to step 2.

## Installation

To install the required packages, run the following command:

```bash
conda env create -f environment.yml
conda acrivate aruco
```
This will create a new conda environment called `aruco` and install the required packages. You can activate the environment by running `conda activate aruco`.
<br>

## Usage

To estimate the 3D angles of the distorted ArUco markers in real-time, run the following command:

```bash
python aruco_realtime.py
```
The program will launch a window that shows a live video feed, within which it will highlight detected ArUco markers. Plese modify the `camera_matrix` and `dist_coeff` variables through calibrating your own camera parameters.
You can just press the 'q' key to terminate the program.

<br>

To generate ArUco markers, run the following command:

```bash
python aruco_generator.py
```
Please modify the `aruco_type` variable in the script to generate different types of ArUco markers.

<br>

To preprocess(bluring and distorting) the generated ArUco markers from recorded .avi, run the following command:

```bash
python aruco_preprocess.py
```

<br>

To detect ArUco markers and estimate their 3D angles from preprocessed ArUco markers, run the following command:

```bash
python aruco_pose_est.py
```
