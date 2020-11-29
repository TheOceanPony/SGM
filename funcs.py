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


@njit
def init_unary_penalty_matrix(line_L, line_R, width, max_disp):

    H = np.zeros((width, max_disp+1), dtype=np.uint8)

    for i in range(0, width):
        for di in range(0, max_disp+1):
            if True: # i >= di:
                H[i, di] = abs(line_L[i] - line_R[i-di])  # If no @njit -- strange warning...
            #else:
                #H[i, di] = 0

    return H


# Left part
@njit
def init_left_part(width, max_disp, H, G):

    Li = np.zeros((width, max_disp+1), dtype=np.float32)

    for i in range(0, width):
        for d in range(0, max_disp+1):
            Li[i, d] = left(i, d, max_disp, Li, H, G)

    return Li

@njit
def left(i, d, max_disp, Li, H, G):

    if i == 0:
        return 0
    else:
        minl = np.inf  # float("infinity") -- doesn't work with numba

        for dj in range(0, max_disp + 1):
            temp = Li[i - 1, dj] + H[i-1, dj] + G[d, dj]
            if temp < minl:
                minl = temp

        return minl


# Right part
@njit
def init_right_part(width, max_disp, H, G):

    Ri = np.zeros((width, max_disp+1), dtype=np.float32)

    for i in range(0, width):
        for d in range(0, max_disp+1):
            Ri[i, d] = right(i, d, max_disp, Ri, H, G)

    return Ri


@njit
def right(i, d, max_disp, Ri, H, G):

    if i == Ri.shape[0]:
        return 0
    else:
        minr = np.inf  # float("infinity") -- doesn't work with numba

        for dj in range(0, max_disp + 1):
            temp = Ri[i + 1, dj] + H[i+1, dj] + G[d, dj]
            if temp < minr:
                minr = temp

        return minr

@njit
def init_previous_index_matrix(width, max_disp, Fi, G):

    Pi = np.zeros((width, max_disp+1), dtype=np.uint8)

    Pi[0,:] = 0


    for i in range(0, width):
        for d in range(0, max_disp+1):

            # argmin_dj ( f_i-1(dj) + g(d, dj) )
            minf = np.inf
            mind = 0

            for dj in range(0, max_disp+1):

                temp = Fi[i-1, dj] + G[dj, d]
                if temp < minf:
                    minf = temp
                    mind = dj

            Pi[i, d] = mind

    return Pi


@njit
def argmin(i, max_disp, Fi):

    minf = 999999
    mind = None

    for d in range(0, max_disp+1):
        temp = Fi[i, d]
        if temp < minf:
            minf = temp
            mind = d

    return mind
