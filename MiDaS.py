import torch
import cv2
import numpy as np
import urllib.request
from torchvision.transforms import Compose

# Load model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Load image
img = cv2.imread("./Middlebury/Right_Cetatii.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Depth estimation
depth_map = midas(torch.tensor(img).unsqueeze(0))

cv2.imshow("Depth Map", depth_map.numpy())
cv2.waitKey(0)
