import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d

def compute_disparity(left_img_path, right_img_path):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,  # ajustat mai mic
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=6,
        speckleWindowSize=100,
        speckleRange=16
    )

    disparity_raw = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # Filtrare bilaterală pentru netezirea rezultatului
    disparity_filtered = cv2.bilateralFilter(disparity_raw, d=15, sigmaColor=75, sigmaSpace=75)

    np.save("disparity_map.npy", disparity_filtered)
    return disparity_filtered, imgL