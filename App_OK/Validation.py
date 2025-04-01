import numpy as np
import matplotlib.pyplot as plt

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

    # Vizualizare comparație
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