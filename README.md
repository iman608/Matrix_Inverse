# Matrix Inverse using Gauss–Jordan Elimination

This project is a **university-level implementation** of matrix inversion using the **Gauss–Jordan elimination method**.
The goal of this project is to demonstrate a clear understanding of linear algebra concepts and elementary row operations.

## Project Description
The program computes the inverse of a square matrix by transforming the original matrix into the identity matrix using elementary row operations.
The same operations are simultaneously applied to an identity matrix to obtain the inverse.

The implementation focuses on correctness, clarity, and numerical stability.

## Algorithm
- Gauss–Jordan elimination
- Elementary row operations:
  - Row swapping
  - Row scaling
  - Row replacement
- Partial pivoting is used when necessary to avoid zero or near-zero pivots

## Features
- Accepts an `n × n` matrix as input
- Detects non-invertible (singular) matrices
- Computes the inverse using NumPy arrays
- Verifies correctness by computing `A⁻¹ × A`, which should result in the identity matrix

## Requirements
- Python 3.13
- NumPy

Install NumPy:
```bash
pip install numpy
```

How to Run
```bash
python matrix_revers.py
```


# Input Format

1.Enter the matrix size n

2.Enter n rows, each containing n integers separated by spaces

# Example:
```bash
enter the rows and cols:
3
1 2 3
0 1 4
5 6 0
```
# Output

•Inverse matrix: the computed inverse

•prodt: verification output (Inverse × Original) — should be close to the identity matrix

# Notes

•Floating-point arithmetic is used, so results may contain small rounding errors.

•If the matrix is singular (non-invertible), the program raises ValueError.

## License
MIT License
