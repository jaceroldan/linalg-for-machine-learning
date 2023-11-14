import numpy as np
from scipy.linalg import qr

def givens_rotation(pivot, target) -> (float, float):
    """
    Retrieves coefficients s and c given a pivot and a target
    for the Givens Rotation algorithm.

    :param pivot: element on the main diagonal
    :param target: item we want to zero
    :return s: Givens Rotation coefficient
    :return c: Givens rotation coefficient
    """
    
    # Base case when the number below the diagonal is already a 0
    if target == 0:
        c, s = 1, 0
    else:
        if abs(target) > abs(pivot):
            r = pivot / target
            s = 1 / np.sqrt(1 + r * r)
            c = s * r
        else:
            r = target / pivot
            c = 1 / np.sqrt(1 + r * r)
            s = c * r
    return c, s


def qr_decomp(A) -> tuple([np.array, np.array]):
    """
    Decompose matrix A into component matrices
    orthogonal matrix Q and upper triangular matrix R
    for which the eigenvalues for A can be extracted
    from R's main diagonal.

    :param A: Matrix to decompose
    :return Q: Orthogonal matrix Q
    :return R: Right triangular matrix R
    """
    n, _ = A.shape
    R = A.copy()
    Q = np.identity(n)

    for _ in range(0, 100):
        q_k = np.identity(n)
        for i in range(n - 1):
            for j in range(i + 1, n):
                c, s = givens_rotation(R[i, i], R[j, i])

                # Givens rotation matrix
                G = np.identity(n)
                G[i, i] = c
                G[j, j] = c
                G[i, j] = -s
                G[j, i] = s

                R = G.T @ R
                q_k = q_k @ G
        
        Q = Q @ q_k
    
    return Q, R


def qr_eigvals(A, tolerance=1e-32, maxiter=1000) -> tuple([np.array, float, int]):
    """
    Retrieve the eigenvalues of A using QR decomposition.

    :param A: matrix input
    :param tolerance: tolerance number to declare 'passable' convergence
    :param maxiter: number of iterations before expected convergence
    """

    A_copy = A.copy()
    A_i = A.copy()

    for _ in range(maxiter):
        A_copy = A_i.copy()
        Q, R = qr_decomp(A_copy)

        A_i = R @ Q

        diff = np.abs(A_i - A_copy).max()

        # Early acceptanble convergence
        if diff < tolerance:
            break

    return np.diag(A_i)


# Example usage
A = np.array([[4, 1, 2], [1, 3, 1], [2, 1, 2]], dtype=float)
# A = np.array([[0, -1, 1], [4,2,0], [3,4,0]], dtype=float)
# A = np.array([[1,2,3],[2,2,4], [3,4,2]], dtype=float)


print(sorted(qr_eigvals(A)))
print(sorted(np.linalg.eigvals(A)))

if np.allclose(A, A.T):
    print("Matrix A is symmetric. Eigenvectors are: ")
    Q, R = qr_decomp(A) 
    print(Q)
    print(np.linalg.eig(A)[1])
    import scipy
    print(scipy.linalg.eig(A)[1])
