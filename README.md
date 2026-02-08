# Matrix Inverse (Gauss–Jordan Elimination) — Python + NumPy

This project computes the inverse of a square matrix using **Gauss–Jordan elimination** (elementary row operations).  
The same row operations are applied to an identity matrix to produce the inverse.

## Features
- Reads an `n × n` matrix from standard input
- Uses partial pivoting if the diagonal pivot is zero (or near zero)
- Detects non-invertible matrices and raises an error
- Prints the inverse matrix and checks the result by printing `A⁻¹ × A`

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
