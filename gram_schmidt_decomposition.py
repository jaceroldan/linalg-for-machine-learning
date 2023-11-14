import math
import numpy as np


def dot(u, v) -> float:
    """ Returns the dot product for 2 vectors u and v

    :param u: vector input
    :param v: vector input
    :return: inner product of these 2 1-d arrays.
    """
    return np.sum([u_col * v_col for u_col, v_col in zip(u,v)])

def norm(v) -> float:
    """ Returns the norm of a vector v

    :param v: vector input
    :return: euclidian norm of the vector v
    """
    return math.sqrt(np.sum([v_comp ** 2 for v_comp in v]))


def gram_schmidt(A) -> [np.array, np.array, np.array]:
    """ Returns the matrices Q, R, and P from the linear equation PA = QR
    through implementation of the Gram-Schmidt algorithm.

    :param A: matrix input for decomposition.
    :return Q: orthogonal matrix Q
    :return R: upper triangular matrix R
    :return P: permutation matrix P for column pivoting
    """

    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    P = np.identity(n)

    # For column pivoting, we look for the pivot column with 
    # the greatest euclidian norm.
    for j in range(n):
        max_norm = 0
        p_index = j

        for i in range(j, n):
            col_norm = norm(A[:, i])
            if col_norm > max_norm:
                max_norm = col_norm
                p_index = i

        # Swap all the columns given the discovered pivot.
        if p_index != j:
            A[:, [j, p_index]] = A[:, [p_index, j]]
            Q[:, [j, p_index]] = Q[:, [p_index, j]]
            P[:, [j, p_index]] = P[:, [p_index, j]]

        # The actual Gram-Schmidt steps start below:

        # Start with the j-th column of A as the j-th column of Q
        # (1. Get ai or the ith column of A)
        Q[:, j] = A[:, j]

        # Make the j-th column of Q orthogonal
        # (2. Project ai to Q to get pi)
        for i in range(j):
            R[i, j] = dot(Q[:, i], Q[:, j])
            # We get the difference between the vector ai
            # and its projection 
            # (3. subtract pi from ai to get ei)
            Q[:, j] -= R[i, j] * Q[:, i]

        # Normalize the j-th column of Q
        # (4. normalize the error)
        # Ensures that the vectors on Q remain orthonormal 
        R[j, j] = norm(Q[:, j])
        if R[j, j] != 0:
            Q[:, j] /= R[j, j]
        else:
            continue

    # We repeat the above steps for all columns ai in A.
    return Q, R, P


###########
# Testing #
###########
A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
# A = np.array([[1,2,-3,1],[2,4,0,7], [-1,3,2,0]], dtype=float)
# A = np.array([[1,2,3],[2,4,6]], dtype=float)
Q, R, P = gram_schmidt(A)

print('Q = ', Q)
print('R = ', R)
print('P = ', P)
print(P @ A)
print(Q @ R)
