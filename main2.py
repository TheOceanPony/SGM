import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import io, color

import funcs2 as f2

if __name__ == '__main__':

    # Params
    MAX_DISP = 50
    alpha = 10

    # Reading imgs
    img_L = (color.rgb2gray( io.imread("imgs/im0.png") )*255)
    img_R = (color.rgb2gray(io.imread("imgs/im1.png")) * 255)

    #img_L = img_L[300:900, 300:900]
    #img_R = img_R[300:900, 300:900]

    height, width = img_L.shape

    print(f"Img info - shape: {width, height}, max el: {np.max(img_L)}, dtype: {img_L.dtype}")
    # Initialize unary penalty matrix
    H = f2.init_unary_penalty_matrix(img_L, img_R, width, height, MAX_DISP)

    # Initialize binary penalty matrix
    G = f2.init_binary_penalty_matrix(MAX_DISP, alpha)

    # Left
    Li = f2.init_left_part(width, height, MAX_DISP, H, G)

    # Right
    Ri = f2.init_right_part(width, height, MAX_DISP, H, G)

    # Top
    Ti = f2.init_top_part(width, height, MAX_DISP, H, G)

    # Bottom
    Bi = f2.init_bottom_part(width, height, MAX_DISP, H, G)

    # Algorithm
    Depthmap = np.zeros((height, width), dtype=np.dtype(float))


    # SGM-Cross
    print('All initialized. Starting depthmap reconstruciton...')
    for j in tqdm(range(0, height)):

        Dm = np.zeros(width, dtype=np.uint8)
        for i in range(0, width):
            best_len, best_d = np.inf, None
            for dt in range(0, MAX_DISP + 1):
                temp = Li[j, i, dt] + Ti[j, i, dt] + H[j, i, dt] + Ri[j, i, dt] + Bi[j, i, dt]
                if temp < best_len:
                    best_len = temp
                    best_d = dt

            Dm[i] = best_d

        Depthmap[j] = Dm

    Depthmap = Depthmap*(255.0/MAX_DISP)
    print(f"dtype: {Depthmap.dtype}, max: {np.max(Depthmap)}")
    plt.imsave("imgs/results/Result-Cross.png", Depthmap, cmap='gray')





