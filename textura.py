import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def run_middlebury_stereo_matching(left_img_path, right_img_path):
    # Obține calea directorului pentru salvare
    save_directory = os.path.dirname(os.path.abspath(__file__))
    print("Aici")
    # Încarcă imaginile stânga și dreapta
    imgL = cv2.imread(left_img_path)
    imgR = cv2.imread(right_img_path)

    # Convertim la grayscale (obligatoriu pentru stereo matching)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Creează obiectul StereoSGBM
    window_size = 5
    min_disp = 0
    num_disp = 128  # Trebuie să fie multiplu de 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=9,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    # Calculează harta de disparitate
    print("🔄 Calculăm harta de disparitate...")
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0  # împărțit la 16 pentru scalare corectă

    # Salvăm datele brute .npy
    np.save(os.path.join(save_directory, "disparity_map.npy"), disparity)

    # Normalizăm pentru vizualizare și salvare ca imagine
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Salvăm imaginea ca PNG
    cv2.imwrite(os.path.join(save_directory, "disparity_map.png"), disp_vis)

    # Afișăm imaginea
    plt.imshow(disp_vis, cmap='magma')
    plt.title("Harta de disparitate (StereoSGBM)")
    plt.axis("off")
    plt.show()

    print("✅ Harta de disparitate a fost salvată cu succes!")


left_image_path = "App_OK/Middlebury/Left_Pendulum.png"
right_image_path = "App_OK/Middlebury/Right_Pendulum.png"
print("aici")
run_middlebury_stereo_matching(left_image_path, right_image_path)