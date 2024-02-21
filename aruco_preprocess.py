import os
import cv2
import numpy as np
from glob import glob


def estimate_blur(img):
    """
    Estimate the sharpness of an image using the Laplacian operator.
    Higher values indicate that the image is sharper.
    :param img: The input image.
    :return: The variance of the Laplacian operator.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def match_blur(ori_img, laplacian_var, ksize=3):
    """
    Adjust the sharpness of img2 to match img1.
    Return the adjusted image.
    :param ori_img: The image to adjust.
    :param laplacian_var: The variance of the Laplacian operator of ori_img.
    :param ksize: The kernel size of the Gaussian blur.
    :return: The adjusted image.
    """
    ori_var = estimate_blur(ori_img)
    print("ori_var", ori_var)

    if ori_var > laplacian_var:
        blurred_img = cv2.GaussianBlur(ori_img, (ksize, ksize), 0)
        blurred_var = estimate_blur(blurred_img)
        print("blurred_var", blurred_var)

    return blurred_img


def apply_rotation(image, rx, ry, rz, scale=1):
    """
    To simulate the effect of a camera with a non-zero roll, pitch, or yaw, 
    we can apply a rotation to the image.
    This function applies a rotation to the image around the x, y, and z axes.
    :param image: The input image.
    :param rx: The rotation around the x-axis in radians.
    :param ry: The rotation around the y-axis in radians.
    :param rz: The rotation around the z-axis in radians.
    :param scale: The scale factor.
    :return: The rotated image.
    """
    h, w = image.shape[:2]
    f = max(h, w)  # Focal length
    cx, cy = w // 2, h // 2
    corners = np.array([[0, 0],
                        [0, h-1],
                        [w-1, h-1],
                        [w-1, 0]], dtype=np.float32)

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))

    M = np.dot(K, np.dot(R, np.linalg.inv(K)))
    M /= M[2, 2]
    M *= scale

    transformed_corners = cv2.perspectiveTransform(corners[None, :, :], M)[0]
    min_x, min_y = transformed_corners.min(axis=0).astype(int) - 10
    max_x, max_y = transformed_corners.max(axis=0).astype(int) + 10
    out_width = max_x - min_x
    out_height = max_y - min_y
    offset_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    M_offset = offset_matrix @ M

    result = cv2.warpPerspective(image,
                                 M_offset,
                                 (out_width, out_height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

    return result


if __name__ == '__main__':

    video_nums = [100, 150, 200, 250, 300, 350,
                  400, 450, 500, 550, 600, 650, 700]

    for num in video_nums:
        video_path = 'recorded/'+str(num)+'_result.avi'
        output_path = 'webcam_images/'+str(num)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open the webcam.")
            exit()

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            output_filename = f'{str(num)}_frame_{frame_count:02d}.png'
            output_filepath = os.path.join(output_path, output_filename)
            cv2.imwrite(output_filepath, frame)

            frame_count += 1
            if frame_count == 5:
                break

        cap.release()
        print(f"{frame_count} images are saved.")

    blur_output_path = 'blurred_markers/'
    if not os.path.exists(blur_output_path):
        os.makedirs(blur_output_path)

    var = 0
    for num in video_nums:

        img_paths = glob("webcam_images/"+str(num)+"/*.png")
        img_num = len(img_paths)

        laplacian_var = 0
        for img_path in img_paths:
            webcam_img = cv2.imread(img_path)
            webcam_var = estimate_blur(webcam_img)
            laplacian_var += webcam_var

        laplacian_var /= img_num
        var += laplacian_var
        print("laplacian_var for", num, ":", laplacian_var)

        ori_img = cv2.imread('makrers/DICT_6X6_250__'+num+'.png')
        blurred_img = match_blur(ori_img, laplacian_var, ksize=3)
        cv2.imwrite(os.path.join(blur_output_path,
                                 'blurred_DICT_6X6_250__'+num+'.png'),
                    blurred_img)

    print("\n\nAverage var:", var/5)

    blurred_imgs = sorted(glob("blurred_markers/*.png"))

    for blurred_img in blurred_imgs:

        marker_size = blurred_img.split('_')[-1][:-4]
        output_path = 'distorted_markers/'+marker_size
        os.makedirs(output_path, exist_ok=True)

        output_img_name = os.path.join(output_path, "distorted_"+marker_size)

        image = cv2.imread(blurred_img)

        angles = []
        angles.append((0, 0, 0))
        for i in range(1, 6):
            angles.append((i/10, 0, 0))
            angles.append((0, i/10, 0))
            angles.append((0, 0, i/10))
            angles.append((-i/10, 0, 0))
            angles.append((0, -i/10, 0))
            angles.append((0, 0, -i/10))

        for idx, (rx, ry, rz) in enumerate(angles):
            rotated = apply_rotation(image, rx, ry, rz)
            cv2.imwrite(output_img_name+f'_{idx:03}_{rx}_{ry}_{rz}.png',
                        rotated)
