import numpy as np


def plu_decomposition(A) -> tuple([np.array, np.array, np.array, int]):
    """
    A: original Matrix

    Returns the P, L, and U matrices as a result of LU
    decomposition on A. This uses partial pivoting (row exchanges).
    It also returns the number of row exchanges.
    """
    m, _ = A.shape
    row_swaps = 0
    
    P = np.identity(m) # create an identity matrix of size m; the permutation matrix
    L = np.identity(m) # Keep track of coefficients of all the row operations; lower triangular matrix
    U = A.copy() # performing row operations (gaussian) on this copy will return U; upper triangular matrix
    
    for x in range(m):
        pivotRow = x

        # search for the best pivot (partial pivot only, however)
        for y in range(x + 1, m, 1):
            if abs(U[y][x]) > abs(U[pivotRow][x]): # Instead of just searching for the first non-zero element
                pivotRow = y
                            
        if U[pivotRow][x] == 0:
            # This column does not have a non-zero element;
            # thus, proceed to next column.
            continue
            
        # If we used a pivot that's not on the diagonal,
        # we do a row exchange and mark the changes on P
        if pivotRow != x:
            # We need to swap the two rows (pivotRow and x) together.
            U[[x, pivotRow]] = U[[pivotRow, x]]
            
            # And since the permutation matrix P mirrors this change
            # we do the same thing but on P!
            P[[x, pivotRow]] = P[[pivotRow, x]]

            row_swaps += 1
            
        # now the pivot row is x; we must 
        # search for rows where the leading coefficient must be eliminated
        for y in range(x + 1, m, 1):
            currentValue = U[y][x]
            if currentValue == 0:
                continue # variable already eliminated, nothing to do, which is awesome.
            
            pivot = U[x][x]
            if pivot == 0:
                # I'm not exactly sure what to do if the pivot is a zero.
                # I'm not sure if it can even get to this part yet, but just hedging against it.
                # Also maybe we can do full pivoting to improve?
                raise Exception('No division by zero.')
            
            pivotFactor = currentValue / pivot
            
            # subtract the pivot row from the current row
            U[[y]] -= pivotFactor * U[[x]]
            
            # L matrix gets the coefficient/pivot factor
            L[y][x] = pivotFactor

    return (P, L, U, row_swaps)


def get_determinant(U, row_swaps) -> float:
    """
    U: the upper triangular matrix
    row_swaps: the number of row_swaps done in the plu_decomposition step

    The calculation of the determinant is done through getting
    the product of the determinants of the U, L, and P matrices. However,
    since L is a lower triangular matrix with all diagonal values 1,
    and P is the permutation matrix, we can get the determinant by
    calculating the product on the diagonal of the U matrix, times the
    product of the diagonal of the L matrix (which is just 1), times
    1 or -1 depending on the number of row exchanges (even or odd, respectively).

    Returns the determinant if the original matrix A is square.
    """
    r, c = U.shape
    if r != c:
        print("This matrix is not square.")
        return 

    # Get the determinant of the U matrix
    determinant = U[0][0]
    for i in range(1, len(U)):
        determinant *= U[i][i]

    # If there were an odd number of row swaps, negate the determinant
    if row_swaps % 2 == 1:
        determinant *= -1

    return determinant 


A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
A = np.array([[1,2,-3,1],[2,4,0,7], [-1,3,2,0]], dtype=float)
A = np.array([[1,1,1], [4,3,-1], [3,5,3]], dtype=float)

P, L, U, row_swaps = plu_decomposition(A)
print('P = ', P)
print('L = ', L)
print('U = ', U)

print('Determinant is: ', get_determinant(U, row_swaps))
