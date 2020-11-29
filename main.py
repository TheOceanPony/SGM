
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage import io, color

import funcs as f

if __name__ == '__main__':

    # Params
    MAX_DISP = 10
    alpha = 10

    # Reading imgs
    img_L = (color.rgb2gray( io.imread("imgs/view0.png") )*255)
    img_R = (color.rgb2gray(io.imread("imgs/view1.png")) * 255)
    height, width = img_L.shape
    width = width - MAX_DISP

    print(f"Img info - shape: {width, height}, max el: {np.max(img_L)}, dtype: {img_L.dtype}")

    # Initialize binary penalty matrix
    G = f.init_binary_penalty_matrix(MAX_DISP, alpha)
    plt.imsave("imgs/results/G.png", G, cmap='gray')

    # Algorithm
    Depthmap = np.zeros((height, width))

    for row in tqdm(range(0, height)):

        line_L = img_L[row, :]
        line_R = img_R[row, :]

        # Initialize unary penalty matrix
        H = f.init_unary_penalty_matrix(line_L, line_R, width, MAX_DISP)

        # Left part
        Li = f.init_left_part(width, MAX_DISP, H, G)
        #plt.imsave("imgs/results/L.png", Li, cmap='gray')

        # Right part
        Ri = f.init_right_part(width, MAX_DISP, H, G)
        #plt.imsave("imgs/results/R.png", Ri, cmap='gray')

        # Reconstructing
        Dm = np.zeros(width, dtype=np.uint8)

        for i in range(0, width):

            best_len, best_d = np.inf, None
            for dt in range(0, MAX_DISP+1):
                temp = Li[i, dt] + H[i, dt] + Ri[i, dt]
                if temp < best_len:
                    best_len = temp
                    best_d = dt

            Dm[i] = best_d

        Depthmap[row] = Dm

    plt.imsave("imgs/results/Result.png", Depthmap, cmap='gray')


