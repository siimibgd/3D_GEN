import cv2
import numpy as np
import os
import open3d as o3d

def load_Q_from_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        if '=' in line:
            key, val = line.strip().split('=')
            key = key.strip()
            val = val.strip().replace('[', '').replace(']', '').replace(';', ' ')  # Eliminăm `;`

            if key.startswith('cam'):  # Matrice de calibrare 3x3
                matrix_values = [float(x) for x in val.split()]  # Conversie corectă
                if len(matrix_values) == 9:
                    params[key] = np.array(matrix_values).reshape(3, 3)
            else:  # Parametrii scalari
                try:
                    params[key] = float(val)
                except ValueError:
                    continue

    # Extrage parametrii necesari
    fx = params['cam0'][0, 0]  # Focal length
    cx = params['cam0'][0, 2]  # Optical center X
    cy = params['cam0'][1, 2]  # Optical center Y
    doffs = params.get('doffs', 0.0)
    baseline = params.get('baseline', 1.0)  # Evităm împărțirea la zero

    # Construim matricea Q
    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1.0 / baseline, doffs / baseline]
    ])
    return Q

def write_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for point, color in zip(points, colors):
            f.write(f"{point[0]} {point[1]} {point[2]} {color[2]} {color[1]} {color[0]}\n")

def generate_3D_pointcloud_with_texture(left_img_path, right_img_path, calib_path):
    # Încarcă imaginile
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Calculează disparitate cu SGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=9,
        P1=8*3*9**2,
        P2=32*3*9**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0

    # Încarcă matricea de calibrare Q
    Q = load_Q_from_calib(calib_path)

    # Reconstrucție 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0
    colors = imgL[mask]  # Folosim imaginea stângă ca textură
    output_points = points_3D[mask]

    # Salvăm norul de puncte texturat
    write_ply("textured_point_cloud.ply", output_points, colors)
    print("✅ Nor de puncte cu textură salvat în textured_point_cloud.ply")
    pcd = o3d.io.read_point_cloud("textured_point_cloud.ply")

    # Afișează modelul
    o3d.visualization.draw_geometries([pcd])
# Exemplu de rulare

left_img = "Middlebury/Left_Chess.png"
right_img = "Middlebury/Right_Chess.png"
calib_file = "calibs/calib_chess.txt"
generate_3D_pointcloud_with_texture(left_img, right_img, calib_file)