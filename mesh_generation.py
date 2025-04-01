import open3d as o3d
import numpy as np
import cv2

# =======================
# 1. Setări cameră stângă
# =======================
K = np.array([[700, 0, 320],   # fx, 0, cx
              [0, 700, 240],   # 0, fy, cy
              [0,   0,   1]])  # 0,  0,  1

width, height = 640, 480  # dimensiunea imaginii stânga
texture_image_path = "App_OK/Middlebury/Left_Pendulum.png"

# =======================
# 2. Încarcă mesh-ul 3D
# =======================
mesh = o3d.io.read_triangle_mesh("mesh.ply")
mesh.compute_vertex_normals()

# =======================
# 3. Proiectează fiecare vertex 3D în imagine
# =======================
vertices = np.asarray(mesh.vertices)
uv_coords = []

for v in vertices:
    v_cam = v.reshape(3, 1)
    proj = K @ v_cam
    x, y = proj[0][0] / proj[2][0], proj[1][0] / proj[2][0]

    # Normalizează UV (între 0 și 1)
    u = x / width
    v = 1.0 - (y / height)  # coordonata v este de jos în sus
    uv_coords.append([u, v])

uv_coords = np.array(uv_coords)

# =======================
# 4. Setează UV în Open3D mesh
# =======================
mesh.triangle_uvs = o3d.utility.Vector2dVector(np.repeat(uv_coords, 3, axis=0))
mesh.textures = [o3d.io.read_image(texture_image_path)]

# =======================
# 5. Salvează mesh-ul în format OBJ + MTL + imagine
# =======================
mesh = o3d.geometry.TriangleMesh()

# Asigură-te că are coordonate UV pentru textură
if mesh.has_vertex_normals() and mesh.has_triangle_uvs():
    print("✅ Mesh-ul conține UVs")

# Salvează mesh-ul texturat
o3d.io.write_triangle_mesh("textured_output.obj", mesh, write_triangle_uvs=True)

# Generează automat un fișier .mtl
with open("textured_output.mtl", "w") as f:
    f.write("newmtl material_0\n")
    f.write("Ka 1.000 1.000 1.000\n")
    f.write("Kd 1.000 1.000 1.000\n")
    f.write("Ks 0.000 0.000 0.000\n")
    f.write("d 1.0\n")
    f.write("illum 2\n")
    f.write("map_Kd texture.png\n")  # Asigură-te că textura există în același folder
mesh = o3d.io.read_triangle_mesh("textured_output.obj")
o3d.visualization.draw_geometries([mesh])
print("✅ Mesh exportat cu UV mapping real către 'textured_output.obj'")