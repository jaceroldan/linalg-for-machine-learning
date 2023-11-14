import math
import numpy as np


def norm(v) -> float:
    """ Returns the norm of a vector v

    :param v: vector input
    :return: euclidian norm of the vector v
    """
    return math.sqrt(np.sum([v_comp ** 2 for v_comp in v]))


def hessenberg_householder(A) -> list:
    """
    Returns a hessenberg matrix H as the result of
    a transformation of the original n by n matrix 
    through Householder reflections.

    :param A: an n by n matrix.
    :return H: the hessenberg matrix.
    """

    num_row, num_col = A.shape
    q = A.copy()
    if num_row != num_col:
        print('Matrix is not square, returning A')
        return q
    
    for i in range(num_col - 1):
        x = q[i+1:, i]
        magnitude = norm(x)

        w = np.zeros((num_row-(i+1), 1), dtype=float)
        x = x.reshape(w.shape)
        w[0][0] = magnitude if x[0][0] <= 0 else magnitude * -1

        v = w - x

        v_times_v_t = v @ v.T
        v_t_times_v = v.T @ v
        P = v_times_v_t / v_t_times_v

        identity = np.eye(v_times_v_t.shape[0])
        h_hat = identity - 2 * P

        H = np.eye(A.shape[0])
        H[i+1:, i+1:] = h_hat
        q = (H @ q @ H).round(6)

    return q


# A = np.asarray([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
# A = np.asarray([[-1,-1,1],[1,3,3],[-1,-1,5]], dtype=float)
A = np.asarray([[1,0,2,3,6],[-1,0,5,2,-1],[2,-2,0,0,3],[2,-1,2,0,1],[-4,-1,2,5,4]], dtype=float)
A = np.asarray([[1,0,2,3],[-1,0,5,2],[2,-2,0,0],[2,-1,2,0]], dtype=float)

print("H = ", hessenberg_householder(A))
