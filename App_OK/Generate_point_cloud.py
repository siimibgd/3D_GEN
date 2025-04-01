import cv2

def generate_point_cloud(disparity, image, Q):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    mask = disparity > 0
    output_points = points_3D[mask]
    output_colors = image[mask]
    return output_points, output_colors

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
