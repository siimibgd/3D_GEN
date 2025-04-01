import cv2
import numpy as np

# Încarcă imaginile stereo (presupunând că sunt deja rectificate)
imgL = cv2.imread('App_OK/Middlebury/Left_Pendulum.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('App_OK/Middlebury/Right_Pendulum.png', cv2.IMREAD_GRAYSCALE)

# Creează un obiect pentru potrivirea stereo (SGBM este un algoritm popular)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 5,  # Numărul de disparități (trebuie ajustat)
    blockSize=15,
    P1=8 * 3 * 15**2,
    P2=32 * 3 * 15**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Calculează harta de disparitate
disparity = stereo.compute(imgL, imgR)

# Normalizare harta de disparitate pentru vizualizare
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Afișează harta de disparitate
cv2.imshow('Harta de disparitate', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Harta de disparitate poate fi apoi utilizată pentru a calcula coordonatele 3D
# (acest pas necesită cunoașterea parametrilor camerei)