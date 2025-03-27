import numpy as np
import open3d as o3d
import os
from PIL import Image

def run():
    os.environ["OPEN3D_HEADLESS"] = "1"

    save_directory = os.path.dirname(os.path.abspath(__file__))
    fused_depth_path = os.path.join(save_directory, "right_depth.npy")

    if not os.path.exists(fused_depth_path):
        print("❌ ERROR: No fused depth map found! Run 'depth_estimation.py' first.")
        exit()

    # Load fused depth map
    fused_depth = np.load(fused_depth_path)

    def depth_to_point_cloud(depth_map, scale=0.1):
        """Converts a depth map to a 3D point cloud WITHOUT texture."""
        h, w = depth_map.shape
        fx = fy = 700
        cx, cy = w // 2, h // 2

        points = []
        for v in range(h):
            for u in range(w):
                Z = depth_map[v, u] * scale
                if Z > 0:
                    X = (u - cx) * Z / fx
                    Y = -(v - cy) * Z / fy
                    points.append([X, Y, Z])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    # Generate point cloud without color
    pcd = depth_to_point_cloud(fused_depth)

    # Save and visualize
    o3d.io.write_point_cloud(os.path.join(save_directory, "point_cloud_no_texture.ply"), pcd)
    o3d.visualization.draw_geometries([pcd])
    print("✅ 3D Point Cloud (NO TEXTURE) saved!")
run()