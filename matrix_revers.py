import numpy as np
import sys

# In this code we read a square matrix from input and use three elementary
# row operations to reduce it to the identity matrix. We apply the same
# operations in the same order to the identity matrix to obtain the inverse.

def cinMatrix(): # Read a matrix; since it should be square, a single number is used for rows and columns
        matrix = []
        size = int(input("enter the rows and cols:"))
    
        for i in range(size):
            row = list(map(int, input().split()))
            matrix.append(row)
        
        return matrix
    
def coutMatrix(M, message="Matrix"):
    print(f"{message}:\n")
    n = M.shape[0]
    for r, row in enumerate(M):
        left  = "⌈" if r == 0     else ("⌊" if r == n-1 else "|")
        right = "⌉" if r == 0     else ("⌋" if r == n-1 else "|")
        line = "  ".join(f"{val:8.3f}" for val in row)
        print(f"{left}{line} {right}")
    print()

        
    print()

def multyMatrix(matrix1, matrix2):  # Multiply two matrices
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError("in multiply the cols in the first matrix dont match the rows in the secound matrix.")
    return matrix1 @ matrix2
    
    

def operation1(m, r1, r2): # Elementary row operation type 1: swap two rows
    m[[r1,r2]] = m[[r2,r1]]
    return m


def operation2(m, m_row, k): # Elementary row operation type 2: multiply a row by a scalar
    """
    # for i in range(m.shape[0]):
    #     m [i,m_cols] = m [i,m_cols] * k
    # return m
    """
    m[m_row, :] *= k
    return m

def operation3(M, target_row, source_row, factor):# Elementary row operation type 3: replace target_row by target_row - factor * source_row
    """
    target_row - factor * source_row → target_row
    """
    M[target_row, :] -= factor * M[source_row, :]
    return M

# Alternative way to write the third elementary row operation function
"""    
def operation3(matrix, row1, num1, op, row2, num2, f_row): 
    
    copy1 = matrix[row1]
    copy2 = matrix[row2]
    
    for i in range(copy1):
        copy1[i] = copy1[i] * num1
        copy2[i] = copy2[i] * num2
    
    result = []
    
    match op:
        case '+':
            for j in range(copy1):
                result[j] = copy1[j] + copy2[j]
                
        case '-':
            for k in range(copy1):
                result[k] = copy1[k] - copy2[k]
                

    for i in range(copy1):
        matrix[f_row, i] = result[i]
        
    return matrix 
"""


    # These three functions ensure that any operation applied to the input
    # matrix is simultaneously applied to the identity matrix
def scale_row_both(L, R, i, alpha):
    L = operation2(L, i, alpha)
    R = operation2(R, i, alpha)

def swap_rows_both(L, R, i, k):
    L = operation1(L, i, k)
    R = operation1(R, i, k)

def add_row_both(L, R, target_row, source_row, factor):
    L = operation3(L, target_row, source_row, factor)
    R = operation3(R, target_row, source_row, factor)


def compute_inverse(matrix_input):
    """Compute and return the inverse of a square numpy array using
    the same Gauss-Jordan procedure implemented in `main`.
    Raises ValueError if the matrix is not square or not invertible.
    """
    matrix = np.array(matrix_input)
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Matrix must be square to have an inverse.")

    L = matrix.astype(float).copy()
    I = np.eye(L.shape[0], dtype=float)
    EPS = 1e-12

    for i in range(matrix.shape[0]):

        if abs(L[i,i]) < EPS:
            pivot_row = np.argmax(abs(L[i: , i])) + i
            if abs(L[pivot_row, i]) < EPS:
                raise ValueError("this matrix dose not have an Inverse matrix.")
            # swap in both L and I
            L[[i,pivot_row]] = L[[pivot_row,i]]
            I[[i,pivot_row]] = I[[pivot_row,i]]

        pivot = L[i,i]
        if abs(pivot) < EPS:
            raise ValueError("this matrix dose not have an Inverse matrix.")

        # scale pivot row
        L[i, :] *= 1.0 / pivot
        I[i, :] *= 1.0 / pivot

        for j in range(matrix.shape[0]):
            if i == j:
                continue
            if abs(L[j, i]) > EPS:
                factor = L[j, i]
                L[j, :] -= factor * L[i, :]
                I[j, :] -= factor * I[i, :]

    if not np.allclose(L, np.eye(L.shape[0]), atol=1e-9):
        raise ValueError("this matrix dose not have an Inverse matrix.")

    return I

def main():
    A = cinMatrix()
    matrix = np.array(A)
    I = compute_inverse(matrix)
    coutMatrix(I, "Inverse matrix")
    coutMatrix((I @ matrix), "prodt")


if __name__ == "__main__":
    main()
        
        
    
    
    
    
    
    
        
