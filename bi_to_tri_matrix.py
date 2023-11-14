import numpy as np

def householder_reflection(v):
    """
    Compute the Householder reflection matrix for a given vector v.
    """
    v = v.reshape(-1, 1)  # Ensure v is a column vector
    n = len(v)
    I = np.identity(n)
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        H = I  # Identity matrix for a zero vector
    else:
        u = v / v_norm
        H = I - 2 * np.dot(u, u.T)
    return H

def bidiagonal_to_tridiagonal(B):
    """
    Transform a bidiagonal matrix into a tridiagonal matrix with a permutation matrix.
    """
    m, n = B.shape
    if m < n:
        P = np.identity(n)
        lim = m
    else:
        P = np.identity(m)
        lim = n

    T = B.copy()  # Initialize the tridiagonal matrix

    for i in range(lim - 1):
        # Choose the Householder vector to zero out the subdiagonal element
        v = T[i + 1:, i]
        H = householder_reflection(v)

        # Update both T and P
        T[i+1:, i:] = np.dot(H, T[i+1:, i:])
        T[:, i+1:] = np.dot(T[:, i+1:], H.T)
        P = np.dot(P, H.T)

    return T, P

# Example usage:
B = np.array([[1, 2, 0, 0],
              [0, 3, 4, 0],
              [0, 0, 5, 6]])

tridiagonal, permutation = bidiagonal_to_tridiagonal(B)

print("Bidiagonal Matrix (B):")
print(B)
print("\nTridiagonal Matrix (T):")
print(tridiagonal)
print("\nPermutation Matrix (P):")
print(permutation)
