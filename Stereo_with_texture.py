import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import open3d as o3d

# ==================== Step 1: Compute disparity map ====================
def compute_disparity(left_img_path, right_img_path):
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=7,
        P1=8 * 3 * 7 ** 2,
        P2=32 * 3 * 7 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=16
    )

    disparity_raw = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disparity_filtered = cv2.bilateralFilter(disparity_raw, d=9, sigmaColor=75, sigmaSpace=75)

    np.save("disparity_map.npy", disparity_filtered)
    return disparity_filtered, imgL

# ==================== Step 2: Load calibration and compute Q ====================
def load_Q_from_calib(calib_path):
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        if '=' in line:
            key, val = line.strip().split('=')
            key = key.strip()
            val = val.strip().replace('[', '').replace(']', '')
            if key.startswith('cam'):
                arr = [float(x) for x in val.replace(';', '').split()]
                matrix = np.array(arr).reshape(3, 3 if len(arr) == 9 else 4)
                params[key] = matrix
            else:
                try:
                    params[key] = float(val)
                except:
                    continue

    fx = params['cam0'][0, 0]
    cx = params['cam0'][0, 2]
    cy = params['cam0'][1, 2]
    doffs = params.get('doffs', 0.0)
    if doffs == 0:
        doffs = 1.0

    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1.0 / doffs, 0]
    ])
    return Q

# ==================== Step 3: Reproject to 3D ====================
def generate_point_cloud(disparity, image, Q):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0
    output_points = points_3D[mask]
    output_colors = image[mask]
    return output_points, output_colors

# ==================== Step 4: Save and show point cloud ====================
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

def show_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)
    o3d.visualization.draw_geometries([pcd])

# ==================== Step 5: Read .pfm files ====================
def read_pfm(file):
    with open(file, "rb") as f:
        header = f.readline().rstrip().decode('utf-8')
        color = header == "PF"
        dim_line = f.readline().decode("utf-8")
        while dim_line.startswith("#") or dim_line.strip() == "":
            dim_line = f.readline().decode("utf-8")
        width, height = map(int, dim_line.strip().split())
        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape)[::-1]

# ==================== Step 6: Compare with ground truth ====================
def validate_disparity(my_disp, gt_disp_path):
    gt_disp = read_pfm(gt_disp_path).astype(np.float32)
    gt_disp[gt_disp == 0] = np.nan
    my_disp[my_disp <= 0] = np.nan

    error = np.abs(my_disp - gt_disp)
    valid = ~np.isnan(gt_disp)
    avg_error = np.nanmean(error[valid])
    print(f"\n📏 Mean absolute disparity error: {avg_error:.2f} pixels")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(my_disp, cmap='plasma')
    plt.title("Disparitate estimată")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(gt_disp, cmap='plasma')
    plt.title("Disparitate ground truth")
    plt.axis('off')
    plt.show()

# ==================== MAIN ====================
#if _name_ == "_main_":
left_path = "App_OK/Middlebury/Left_Pendulum.png"
right_path = "App_OK/Middlebury/Right_Pendulum.png"
calib_path = "calibs/calib_pendulum.txt"
gt_disp_path = "disps/disp0_pendulum.pfm"

disparity, imgL = compute_disparity(left_path, right_path)
Q = load_Q_from_calib(calib_path)
points, colors = generate_point_cloud(disparity, imgL, Q)

write_ply("textured_point_cloud.ply", points, colors)
print("✅ Nor de puncte texturat salvat ca textured_point_cloud.ply")

print("\n🔍 Afișăm norul de puncte FĂRĂ textură...")
show_point_cloud(points)

print("\n🎨 Afișăm norul de puncte CU textură din imaginea stângă...")
show_point_cloud(points, colors)

if os.path.exists(gt_disp_path):
    validate_disparity(disparity, gt_disp_path)