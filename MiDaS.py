import open3d as o3d
import numpy as np
import cv2
import os

# Încarcă norul de puncte cu culoare (obținut din imaginea stângă)
pcd = o3d.io.read_point_cloud("point_cloud.ply")
print("✅ Norul de puncte a fost încărcat")

# Estimează normalele
print("🔄 Estimăm normalele...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Downsampling (opțional, pentru performanță)
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# Generare mesh cu Poisson (alternativ la Ball Pivoting)
print("🔄 Generăm mesh-ul (Poisson Reconstruction)...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
print("✅ Mesh generat cu succes!")

# Crop la bounding box-ul norului de puncte (pentru a elimina mesh-uri în afara scenei)
bbox = pcd.get_axis_aligned_bounding_box()
mesh = mesh.crop(bbox)

# Aplică textura: convertim imaginea într-un mesh texturat manual
# În acest caz, salvăm doar culorile punctelor pe mesh-ul triangulat

# Vizualizare mesh + culoare din point cloud
print("🎨 Aplicăm culorile din point cloud pe mesh...")
mesh.vertex_colors = pcd.colors  # simplu, dar eficient

# Salvează și vizualizează
o3d.io.write_triangle_mesh("textured_mesh.ply", mesh)
print("✅ Mesh salvat ca textured_mesh.ply")
o3d.visualization.draw_geometries([mesh])