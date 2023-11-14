import numpy as np

# Define your matrix A
A = np.array([[1, 4, 5],
              [3, 2, 5],
              [2, 1, 3]])

# Perform Gaussian elimination manually to get row-echelon form of A
def gaussian_elimination(matrix):
    m, n = matrix.shape
    r = 0  # Current row
    for c in range(n):  # For each column
        # Find the first row with a non-zero entry in the current column
        for r2 in range(r, m):
            if matrix[r2, c] != 0:
                break
        else:
            # No non-zero entry in this column, move to the next column
            continue
        
        # Swap the rows to make the pivot the current row
        matrix[[r, r2]] = matrix[[r2, r]]
        
        # Normalize the pivot row
        matrix[r] = matrix[r] / matrix[r, c]
        
        # Eliminate non-zero entries below the pivot
        for r2 in range(r + 1, m):
            matrix[r2] -= matrix[r] * matrix[r2, c]
        
        r += 1  # Move to the next row
    
    return matrix

# Perform Gaussian elimination on matrix A
row_echelon_form_A = gaussian_elimination(A.copy())

# Find the pivot rows (rows with leading non-zero entries) in the row-echelon form
pivot_rows = np.unique(np.where(row_echelon_form_A[:, :-1] != 0)[0])

# The basis of the row space of A consists of the corresponding rows in the original matrix A
basis_row_space = A[pivot_rows]

print("Basis of the Row Space of A:")
print(basis_row_space)
