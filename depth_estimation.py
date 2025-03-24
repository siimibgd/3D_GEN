import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import os

# Ensure we save the files in the same directory as this script
save_directory = os.path.dirname(os.path.abspath(__file__))

# Load the Depth Anything model
model_path = "./depth-anything-large-hf"
processor = AutoImageProcessor.from_pretrained(model_path,use_fast=True)
model = AutoModelForDepthEstimation.from_pretrained(model_path)

def load_and_preprocess(image_path):
    """Loads and preprocesses an image for depth estimation."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return image, inputs

# Load left and right images
left_img_path = "Left.jpeg"
right_img_path = "Right.jpeg"

left_image, left_inputs = load_and_preprocess(left_img_path)
right_image, right_inputs = load_and_preprocess(right_img_path)

# Run inference on both images
with torch.no_grad():
    left_output = model(**left_inputs)
    right_output = model(**right_inputs)

# Extract depth predictions
left_depth = left_output.predicted_depth.squeeze().cpu().numpy()
right_depth = right_output.predicted_depth.squeeze().cpu().numpy()

# Save depth maps
np.save(os.path.join(save_directory, "left_depth.npy"), left_depth)
np.save(os.path.join(save_directory, "right_depth.npy"), right_depth)

# Normalize depth maps for visualization
def normalize_depth(depth_map):
    return cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

left_depth_vis = normalize_depth(left_depth)
right_depth_vis = normalize_depth(right_depth)

cv2.imwrite(os.path.join(save_directory, "left_depth.png"), left_depth_vis)
cv2.imwrite(os.path.join(save_directory, "right_depth.png"), right_depth_vis)

# Display both depth maps
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(left_depth_vis, cmap='magma')
axes[0].set_title("Left Image Depth")
axes[0].axis("off")

axes[1].imshow(right_depth_vis, cmap='magma')
axes[1].set_title("Right Image Depth")
axes[1].axis("off")

plt.show()

print("✅ Depth maps saved for both left and right images!")
