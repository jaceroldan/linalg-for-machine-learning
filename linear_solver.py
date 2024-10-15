import numpy as np
from numpy.linalg import solve

# Perform Gaussian elimination manually to get row-echelon form
def gaussian_elimination(A):
    """
    @params: a matrix A
    @return: row echelon form of the matrix, A
    @return: number of independent columns, pivot
    """
    m, n = A.shape
    row = 0
    pivot = 0
    for c in range(n):
        for row2 in range(row, m):
            if A[row2, c] != 0:
                pivot += 1
                break
        else:
            continue

        A[[row, row2]] = A[[row2, row]]
        
        A[row] = A[row] / A[row, c]
        
        for row2 in range(row + 1, m):
            A[row2] -= A[row] * A[row2, c]
        
        row += 1
    
    return A, pivot


def solve_for_r(A, C):
    """
    @params: Two matrices A and C of the decomposition
    A = CR.

    @return: R
    """

    result = np.zeros((C.shape[1], A.shape[1]))
    for i in range(0, A.shape[1]):
        curr_sol = solve(C[:C.shape[0]-1], A[:C.shape[0]-1, i])
        result[:,i] = curr_sol

    return result


def run_solution(A):
    _, pivot = gaussian_elimination(A.copy())
    independent_columns = A[:, :pivot]
    print("C: ")
    print(independent_columns)
    print("R: ")
    print(solve_for_r(A.copy(), independent_columns.copy()))


if __name__ == "__main__":
    # Define your matrix A
    A = np.array([[1, 2, 1],
                  [1, -1, 1],
                  [-1, 1, -1]])

    # A = np.array([[1, 2, 3, 0],
    #               [1, -1, 0, -3],
    #               [-1, 1, 0, 3]])
    run_solution(A)
