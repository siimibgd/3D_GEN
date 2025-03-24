import open3d as o3d

# Încarcă mesh-ul 3D
mesh = o3d.io.read_triangle_mesh("mesh.ply")

# Verifică dacă mesh-ul are coordonate UV
if not mesh.has_triangle_uvs():
    print("Mesh-ul nu are coordonate UV! Va fi necesar să le generezi sau să le imporți.")
    # Poți încerca să generezi UVs manual sau să folosești o aplicație externă pentru asta (de exemplu Blender)

# Încărcăm textura (imaginea)
texture_image_path = "FlowerSet001.png"
texture = o3d.io.read_image(texture_image_path)

# Aplică textura pe mesh
mesh.textures = [texture]

# Vizualizează mesh-ul cu textura aplicată
o3d.visualization.draw_geometries([mesh], window_name="Textured Mesh")
