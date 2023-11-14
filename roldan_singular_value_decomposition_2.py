# TODO 1. Improve algorithm 1A
# There is a problem with matrices
# that are wider than they are tall

import numpy as np
import math

np.set_printoptions(linewidth=400)

def norm(v: np.array) -> float:
    """
    Returns Euclidean norm of the matrix.

    :param v: vector
    """
    return math.sqrt(np.sum([v_comp ** 2 for v_comp in v]))


def householder_reflection(x: np.array, i: int) -> np.array:
    """
    Returns matrix P that will allow us
    to zero out items below or to the right
    of current pivot (depending on supplied vector x
    and usage from caller).

    :param x: vector
    :param i: index containing pivot
    :return P: Householder Matrix
    """
    e = np.zeros(len(x))
    e[i] = norm(x) * (1 if x[i] < 0 else -1)

    v = x - e
    w = v / norm(v)
    P = np.identity(len(x)) - 2 * np.outer(w, w.T)
    
    return P


def bidiagonalize(A: np.array) -> np.array:
    """
    Returns matrix J that is a bidiagonal matrix
    transformed from matrix A through symmetric
    householder reflections above and below the
    diagonal. (Golub-Kahan algorithm)

    This should be algorithm 1A for the source outlined in
    https://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
    (Householder reflections to bidiagonal form)

    :param A: matrix
    :return J: bidiagonal matrix
    :return U: final product of householder matrices from the left
    :return V: final product of householder matrices from the right
    """
    J = A.copy()

    # Find the shape of the input matrix and 
    # find the lower-magnitude dimension and
    # limit the householder algorithm to that
    # bound.
    row, col = A.shape    
    iters = row if row < col else col
    U = np.identity(row)
    V = np.identity(col)

    for i in range(iters):
        # row reflection: causes all elements below the pivot to zero out.
        if i < row - 1:
            h = np.zeros(len(J[:, i]))
            h[i:] = J[i:, i]
            P = householder_reflection(h, i)
            J = (P @ J)
            U = U @ P

        # column reflection: causes all elements to the right of the pivot to zero out
        if i < col - 1:
            h = np.zeros(len(J[i, :]))
            h[i+1:] = J[i, i+1:]
            Q = householder_reflection(h, i+1)
            J = (J @ Q)
            V = Q @ V

    return J[0:i+1, 0:i+1].round(6), U, V


def tridiagonalize(J: np.array) -> np.array:
    """
    Returns a matrix S that is transformed from the form

    M = [
        [0,   J]
        [J.T, 0]
    ]

    into a tridiagonal matrix S using a permutation matrix P
    through the product S = P * M * (P.T) 

    where P.T is the transpose of P.

    :param J: bidiagonal matrix
    :return S: tridiagonal marix
    """
    shape = J.shape[0]

    # Permutation matrix is set to have dimensions
    # 2n, where (n,n) is the original shape of bidiagonal
    # matrix input J.
    P = np.eye(shape * 2)

    # Swaps original order of P = [1, 2, ..., n, n+1, ..., 2n]
    # into order [n+1, 1, n+2, 2, ..., 2n, n]
    P[[*[i for i in range(0, 2 * shape)]]] = P[[*[i // 2 if i % 2 == 1 else i // 2 + shape for i in range(0, 2 * shape)]]]

    M = np.block([[np.zeros((J.shape[0], J.shape[0])), J], [J.T, np.zeros((J.shape[1], J.shape[1]))]])
    S = P @ M @ P.T
    return S


# Test

# A = np.array([[4, 3, 0, 2, 5], [2, 1, 2, 1, 6], [4, 4, 0, 3, 0], [5, 6, 1, 3, 7]])
A = np.random.rand(5,3)
# A = np.array([[-24, 41, -21], [-71, -96, 1], [92, 12, -93], [-75, -11, 37], [18, -39, -19], [-52, -61, 61], [-40, -87, 69], [-4, -42, -29], [-67, 10, -1], [51, 81, 21]])
J = bidiagonalize(A)
print('J = ')
print(J)
S = tridiagonalize(J)
print('S = ')
print(S)
