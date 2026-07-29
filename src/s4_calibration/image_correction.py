import cv2
import numpy as np
import glob
from src.s2_statistics_data_mining.stats import to_grayscale, compute_contrast

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

def normalize_exposure(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )
    l_corrected = clahe.apply(l)
    lab_corrected = cv2.merge([l_corrected,a,b])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)



if __name__ == "__main__":
    #for calibrate and undistortion
    # camera_matrix, dist_coeffs = calibrate_camera("data/calibration")
    # img = cv2.imread("data/calibration/frame-0.png")
    # corrected = undistort_image(img, camera_matrix, dist_coeffs)
    # combined = np.hstack([img, corrected])  
    # cv2.imwrite("docs/calibration_before_after.png", combined)
    # print("saved docs/calibration_before_after.png")
    #for clahe
    img = cv2.imread("data/VisDrone2019-DET-train/images/0000070_05776_d_0000004.jpg")
    corrected = normalize_exposure(img)
    combined = np.hstack([img, corrected])
    cv2.imwrite("docs/clahe_before_after.png", combined)
    print("saved docs/clahe_before_after.png")
    print("contrast before:", compute_contrast(to_grayscale(img)))
    print("contrast after: ", compute_contrast(to_grayscale(corrected)))