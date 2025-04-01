import numpy as np
import open3d as o3d
import os
from PIL import Image

def run(paths):
#def run():
    os.environ["OPEN3D_HEADLESS"] = "1"

    save_directory = os.path.dirname(os.path.abspath(__file__))
    fused_depth_path = os.path.join(save_directory, "disparity_map.npy")
    #left_image_path = os.path.join(save_directory, paths[1])
    left_image_path = os.path.join(save_directory, paths[0])

    if not os.path.exists(fused_depth_path):
        print("❌ ERROR: No fused depth map found! Run 'depth_estimation.py' first.")
        exit()

    # Load fused depth map and left image
    fused_depth = np.load(fused_depth_path)
    left_image = np.array(Image.open(left_image_path))


    def depth_to_point_cloud(depth_map, image, scale=0.1):
        """Converts a depth map to a 3D point cloud with colors and corrects mirroring."""
        h, w = depth_map.shape
        fx = fy = 700
        cx, cy = w // 2, h // 2

        points = []
        colors = []
        for v in range(h):
            for u in range(w):
                Z = depth_map[v, u] * scale
                if Z > 0:
                    X = (u - cx) * Z / fx  # Corectare a poziției X
                    Y = -(v - cy) * Z / fy  # Inversare Y pentru corectarea culorilor
                    points.append([X, Y, Z])

                    # Asigurare că indexul rămâne în limitele imaginii
                    img_u = min(max(u, 0), w - 1)
                    img_v = min(max(v, 0), h - 1)
                    colors.append(image[img_v, img_u] / 255.0)  # Normalize color values

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd


    pcd = depth_to_point_cloud(fused_depth, left_image)
    o3d.io.write_point_cloud(os.path.join(save_directory, "point_cloud.ply"), pcd)
    o3d.visualization.draw_geometries([pcd])
    print("✅ 3D Point Cloud with color saved!")
#run()