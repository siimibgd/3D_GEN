import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Load depth maps
save_directory = os.path.dirname(os.path.abspath(__file__))
left_depth = np.load(os.path.join(save_directory, "left_depth.npy"))
right_depth = np.load(os.path.join(save_directory, "right_depth.npy"))

# Fusion: Simple Averaging (can be replaced with weighted fusion)
fused_depth = (left_depth + right_depth) / 2

# Save fused depth map
np.save(os.path.join(save_directory, "fused_depth.npy"), fused_depth)

# Normalize for visualization
fused_depth_vis = cv2.normalize(fused_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(os.path.join(save_directory, "fused_depth.png"), fused_depth_vis)

# Display fused depth map
plt.imshow(fused_depth_vis, cmap='magma')
plt.colorbar(label="Depth")
plt.title("Fused Depth Map")
plt.axis("off")
plt.show()

print("✅ Fused depth map saved as 'fused_depth.npy'")
