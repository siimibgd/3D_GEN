import cv2
import numpy as np
import os

def load_Q_from_calib(calib_path):
    calib_path="./calibs/"+calib_path
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    params = {}
    for line in lines:
        if '=' in line:
            key, val = line.strip().split('=')
            key = key.strip()
            val = val.strip().replace('[', '').replace(']', '')
            if key.startswith('cam'):
                arr = [float(x) for x in val.replace(';', '').split()]
                matrix = np.array(arr).reshape(3, 3 if len(arr) == 9 else 4)
                params[key] = matrix
            else:
                try:
                    params[key] = float(val)
                except:
                    continue

    fx = params['cam0'][0, 0]
    cx = params['cam0'][0, 2]
    cy = params['cam0'][1, 2]
    doffs = params.get('doffs', 0.0)
    if doffs == 0:
        doffs = 1.0  # fallback pentru a evita impartirea la zero

    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1.0 / doffs, 0]
    ])
    return Q