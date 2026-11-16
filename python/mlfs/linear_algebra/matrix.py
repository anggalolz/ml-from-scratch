from ctypes import *
import numpy as np
import pathlib

libname = pathlib.Path().absolute() / "build" / "lib" / "matrix.so"
c_lib = CDLL(libname)

class Matrix(Structure):
    _fields_ = [
        ("rows", c_size_t),
        ("cols", c_size_t),
        ("data", POINTER(c_float))
    ]

def mat_add(A, B):
    dim_a = np.shape(A)
    dim_b = np.shape(B)
    if (dim_a[0] != dim_b[0] or dim_a[1] != dim_b[1]):
        raise ValueError
    
    matA = prepare_matrix(A)
    matB = prepare_matrix(B)

    c_lib.mat_add.argtypes = [POINTER(Matrix), POINTER(Matrix)]
    c_lib.mat_add.restype = POINTER(Matrix)

    mat_out = c_lib.mat_add(byref(matA), byref(matB)).contents
    return clean_output_matrix(mat_out)

def mat_sub(A, B):
    dim_a = np.shape(A)
    dim_b = np.shape(B)
    if (dim_a[0] != dim_b[0] or dim_a[1] != dim_b[1]):
        raise ValueError
    
    matA = prepare_matrix(A)
    matB = prepare_matrix(B)

    c_lib.mat_sub.argtypes = [POINTER(Matrix), POINTER(Matrix)]
    c_lib.mat_sub.restype = POINTER(Matrix)

    mat_out = c_lib.mat_sub(byref(matA), byref(matB)).contents
    return clean_output_matrix(mat_out)

def mat_mul(A, B):
    dim_a = np.shape(A)
    dim_b = np.shape(B)
    if (dim_a[1] != dim_b[0]):
        raise ValueError

    matA = prepare_matrix(A)
    matB = prepare_matrix(B)

    c_lib.mat_mul.argtypes = [POINTER(Matrix), POINTER(Matrix)]
    c_lib.mat_mul.restype = POINTER(Matrix)

    mat_out = c_lib.mat_mul(byref(matA), byref(matB)).contents
    return clean_output_matrix(mat_out)

def prepare_matrix(A):
    dim = np.shape(A)
    A = A.flatten()
    A = (c_float * len(A))(*A)
    return Matrix(rows = c_size_t(dim[0]), cols = c_size_t(dim[1]), data = A)

def clean_output_matrix(mat_out):
    rows = mat_out.rows
    cols = mat_out.cols
    return np.array(mat_out.data[:rows*cols]).reshape(rows, cols)

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [3, 7, 10],
    [4, 5, 6]
])

print(A + B)