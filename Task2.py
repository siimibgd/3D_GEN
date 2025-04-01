import cv2
import numpy as np
import open3d as o3d

# Load stereo images
imgL = cv2.imread("App_OK/Middlebury/Left.png", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("App_OK/Middlebury/Right.png", cv2.IMREAD_GRAYSCALE)

# Stereo Matching
stereo = cv2.StereoSGBM_create(numDisparities=64, blockSize=9)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

# Convert to depth
focal_length = 700  # Adjust based on camera
baseline = 0.1  # Distance between cameras
depth_map = (focal_length * baseline) / (disparity + 1e-6)

# Convert depth to point cloud
h, w = depth_map.shape
fx, fy = focal_length, focal_length
cx, cy = w // 2, h // 2

points = []
for v in range(h):
    for u in range(w):
        Z = depth_map[v, u]
        if Z > 0 and Z < 100:  # Filter depth range
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])

# Create Point Cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Save and visualize
o3d.io.write_point_cloud("point_cloud_sgbm.ply", pcd)
o3d.visualization.draw_geometries([pcd])
