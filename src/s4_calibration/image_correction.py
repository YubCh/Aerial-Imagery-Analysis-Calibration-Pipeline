import cv2
import numpy as np
import glob

def calibrate_camera(calibration_images_path, board_size=(9, 6)):
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    object_points = []   
    image_points = []   

    images = glob.glob(f"{calibration_images_path}/*")
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        print(f"{'FOUND ' if found else 'missed'} {fname}")
        if found:
            object_points.append(objp)
            image_points.append(corners)
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, gray.shape[::-1], None, None)
    print("calibration done. distortion coefficients:", dist_coeffs.ravel())
    return camera_matrix, dist_coeffs

def undistort_image(img, camera_matrix, dist_coeffs):
    return cv2.undistort(img, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    camera_matrix, dist_coeffs = calibrate_camera("data/calibration")
    img = cv2.imread("data/calibration/frame-0.png")
    corrected = undistort_image(img, camera_matrix, dist_coeffs)
    combined = np.hstack([img, corrected])  
    cv2.imwrite("docs/calibration_before_after.png", combined)
    print("saved docs/calibration_before_after.png")