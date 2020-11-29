import numpy as np
from numba import njit

# Penalties
@njit
def init_binary_penalty_matrix(max_disp, alpha):
    """
    Binary penalty matrix
    :param max_disp: int
    :param alpha: float
    :return: np.array
    """
    G = np.zeros((max_disp+1, max_disp+1), dtype=np.float32)

    for di in range(0, max_disp+1):
        for dj in range(0, max_disp+1):
            G[di, dj] = alpha*abs(di - dj)

    return G


#@njit
def init_unary_penalty_matrix(img_L, img_R, width, height, max_disp):

    H = np.zeros((height, width, max_disp+1), dtype=np.uint8)

    for j in range(0, height):
        for i in range(0, width):
            for d in range(0, max_disp + 1):
                H[j, i, d] = abs(img_L[j, i] - img_R[j, i - d]) # Not sure about that
    return H


# Horisontal
# Left part
@njit
def init_left_part(width, height, max_disp, H, G):

    Li = np.zeros((height, width, max_disp+1), dtype=np.float32)

    for j in range(0, height):
        for i in range(0, width):
            for d in range(0, max_disp + 1):
                Li[j, i, d] = left(j, i, d, max_disp, Li, H, G)

    return Li

@njit
def left(j, i, d, max_disp, Li, H, G):

    if i == 0:
        return 0
    else:
        minl = np.inf  # float("infinity") -- doesn't work with numba

        for dj in range(0, max_disp + 1):
            temp = Li[j, i - 1, dj] + H[j, i-1, dj] + G[d, dj]
            if temp < minl:
                minl = temp

        return minl


# Right part
@njit
def init_right_part(width, height, max_disp, H, G):

    Ri = np.zeros((height, width, max_disp+1), dtype=np.float32)

    for j in range(0, height):
        for i in range(0, width):
            for d in range(0, max_disp + 1):
                Ri[j, i, d] = right(j, i, d, max_disp, Ri, H, G)

    return Ri

@njit
def right(j, i, d, max_disp, Ri, H, G):

    if i == Ri.shape[0]:
        return 0
    else:
        minr = np.inf  # float("infinity") -- doesn't work with numba

        for dj in range(0, max_disp + 1):
            temp = Ri[j, i + 1, dj] + H[j, i+1, dj] + G[d, dj]
            if temp < minr:
                minr = temp

        return minr


# Vertical
# Top part
@njit
def init_top_part(width, height, max_disp, H, G):

    Ti = np.zeros((height, width, max_disp+1), dtype=np.float32)

    for j in range(0, height):
        for i in range(0, width):
            for d in range(0, max_disp + 1):
                Ti[j, i, d] = top(j, i, d, max_disp, Ti, H, G)

    return Ti

@njit
def top(j, i, d, max_disp, Ti, H, G):

    if j == 0:
        return 0
    else:
        minl = np.inf  # float("infinity") -- doesn't work with numba

        for dj in range(0, max_disp + 1):
            temp = Ti[j-1, i, dj] + H[j-1, i, dj] + G[d, dj]
            if temp < minl:
                minl = temp

        return minl


# Bottom part
@njit
def init_bottom_part(width, height, max_disp, H, G):

    Bi = np.zeros((height, width, max_disp+1), dtype=np.float32)

    for j in range(0, height):
        for i in range(0, width):
            for d in range(0, max_disp + 1):
                Bi[j, i, d] = bottom(j, i, d, max_disp, Bi, H, G)

    return Bi

@njit
def bottom(j, i, d, max_disp, Bi, H, G):

    if i == Bi.shape[0]:
        return 0
    else:
        minr = np.inf  # float("infinity") -- doesn't work with numba

        for dj in range(0, max_disp + 1):
            temp = Bi[j+1, i, dj] + H[j+1, i, dj] + G[d, dj]
            if temp < minr:
                minr = temp

        return minr
