import open3d as o3d
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Asigură-te că scriptul rulează în același director
save_directory = os.path.dirname(os.path.abspath(__file__))
point_cloud_path = os.path.join(save_directory, "point_cloud.ply")
mesh_output_path = os.path.join(save_directory, "mesh.ply")

# Verifică dacă norul de puncte există
if not os.path.exists(point_cloud_path):
    print(f"❌ ERROR: '{point_cloud_path}' not found! Rulează mai întâi 'point_cloud_generation.py'.")
    exit()

# Încarcă norul de puncte
pcd = o3d.io.read_point_cloud(point_cloud_path)
print(f"✅ Norul de puncte a fost încărcat din {point_cloud_path}")

# Estimează normalele (Necesar pentru meshing)
print("🔄 Estimarea normalelor...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
print("✅ Normalele au fost estimate cu succes!")

# Downsampling (Redu densitatea punctelor pentru a accelera procesul)
voxel_size = 0.02  # Setează o dimensiune a voxelului pentru downsampling
print(f"🔄 Reduerea densității punctelor folosind downsampling cu voxel_size={voxel_size}...")
pcd_downsampled = pcd.voxel_down_sample(voxel_size)
print("✅ Downsampling completat!")

# Verifică dimensiunea norului de puncte după downsampling
num_points = np.asarray(pcd_downsampled.points).shape[0]
print(f"🔍 Norul de puncte conține {num_points} puncte după downsampling")

# Verifică dacă numărul de puncte este suficient
if num_points < 100:
    print("⚠️ AVERTISMENT: Norul de puncte are foarte puține puncte! Mesh-ul poate fi gol.")

# Convertește radiile într-un obiect Open3D DoubleVector
radii = o3d.utility.DoubleVector([0.01, 0.02, 0.05])  # Radii optimizate pentru a reduce complexitatea


# Funcția pentru a crea mesh dintr-un subset de puncte
def create_mesh_from_subset(points_chunk):
    pcd_subset = o3d.geometry.PointCloud()
    pcd_subset.points = o3d.utility.Vector3dVector(points_chunk)
    pcd_subset.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_subset, radii)
    return np.asarray(mesh.triangles), np.asarray(mesh.vertices)


# Paralelizează procesul de generare a mesh-ului pe subseturi de puncte
def process_point_cloud(pcd, num_chunks=8):
    # Împarte norul de puncte în mai multe subseturi
    points = np.asarray(pcd.points)
    chunk_size = len(points) // num_chunks
    point_chunks = [points[i:i + chunk_size] for i in range(0, len(points), chunk_size)]

    # Crează un executor pentru a paraleliza procesul
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Aplică algoritmul Ball Pivoting pentru fiecare subset
        results = list(executor.map(create_mesh_from_subset, point_chunks))

    # Combinați triunghiurile și punctele
    all_triangles = np.concatenate([result[0] for result in results], axis=0)
    all_vertices = np.concatenate([result[1] for result in results], axis=0)

    # Crează mesh-ul combinat din triunghiuri și puncte
    combined_mesh = o3d.geometry.TriangleMesh()
    combined_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
    combined_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)

    return combined_mesh


# Protejează secțiunea de multiprocessing cu 'if __name__ == "__main__"'
if __name__ == '__main__':
    # Procesarea norului de puncte pentru a genera mesh-ul
    print("🔄 Crearea mesh-ului din norul de puncte...")
    mesh = process_point_cloud(pcd_downsampled)
    print("✅ Mesh-ul a fost creat cu succes!")

    # Salvează și vizualizează mesh-ul
    o3d.io.write_triangle_mesh(mesh_output_path, mesh)
    print(f"✅ Mesh-ul 3D a fost salvat: {mesh_output_path}")

    # Vizualizează mesh-ul
    o3d.visualization.draw_geometries([mesh])

    # Încărcăm texturile (fișierul de imagine)
    texture_image_path = "Left.jpeg"
    texture = o3d.io.read_image(texture_image_path)
    # Aplică textura pe mesh
    mesh.textures = [texture]
    o3d.visualization.draw_geometries([mesh], window_name="Textured Mesh")


