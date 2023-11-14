import numpy as np
from numpy.linalg import solve

# Define your matrix A
# A = np.array([[1, 2, 3, 4],
#               [4, 5, 6, 6],
#               [7, 8, 9, 7]])

# A = np.array([[1, 2, 3, 0],
#               [1, -1, 0, -3],
#               [-1, 1, 0, 3]])

A = np.array([[1,2,1],
             [1,-1,1],
             [-1,1,-1]])

# A = np.array([[1, 3, 5, 15, 0], [2, 6, 4, 24, 0], [4, 12, 1, 41 ,0]])

# Perform Gaussian elimination manually to get row-echelon form
def gaussian_elimination(matrix):
    m, n = matrix.shape
    r = 0
    pivot = 0
    for c in range(n):
        for r2 in range(r, m):
            if matrix[r2, c] != 0:
                pivot += 1
                break
        else:
            continue

        matrix[[r, r2]] = matrix[[r2, r]]
        
        matrix[r] = matrix[r] / matrix[r, c]
        
        for r2 in range(r + 1, m):
            matrix[r2] -= matrix[r] * matrix[r2, c]
        
        r += 1
    
    return matrix, pivot

print(A.copy())
row_echelon_form, pivot = gaussian_elimination(A.copy())
print('Row Echelon Form: ')
print(row_echelon_form)
print('Solution: ')
# print('Pivot = ', pivot)

independent_columns = A[:, :pivot]
print('Independent columns')
# The C in A = CR
print(independent_columns)


row, col = A.copy().shape

def solve_for_r(A, C):
    """
    Two matrices A and C of the decomposition
    A = CR.

    @return: R
    """
    # shape of R matrix is (# of columns in C x # of columns in A)
    A.shape[1]
    C.shape[1]
    # print('A:', A)
    # print('C:', C)
    # print(A.shape, C.shape)
    result = np.zeros((C.shape[1], A.shape[1]))
    # print(result)
    # print('start')
    # for i in range(0, A.shape[1]):
    #     c_copy = C[:, :]
    #     for j in range(0, A.shape[0]):
    # print(C)
    # print(A[:, 1])
    # print(C[:C.shape[0]-1])
    for i in range(0, A.shape[1]):
        curr_sol = solve(C[:C.shape[0]-1], A[:C.shape[0]-1, i])
        result[:,i] = curr_sol

    # print(solve(A.copy()[:col-1, :col-1], A.copy()[:col-1, col-1:]))
    return result

print(solve_for_r(A.copy(), independent_columns.copy()))
