from ctypes import *
from pathlib import Path
import numpy as np

libname = (
    Path(__file__).resolve().parents[3]
    / "build/lib/matrix.so"
)
c_lib = CDLL(libname)

class CMatrix(Structure):
    _fields_ = [
        ("rows", c_size_t),
        ("cols", c_size_t),
        ("data", POINTER(c_float))
    ]

c_lib.mat_add.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
c_lib.mat_add.restype = POINTER(CMatrix)

c_lib.mat_sub.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
c_lib.mat_sub.restype = POINTER(CMatrix)

c_lib.mat_mul.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
c_lib.mat_mul.restype = POINTER(CMatrix)

c_lib.mat_mul_by_const.argtypes = [c_float, POINTER(CMatrix)]
c_lib.mat_mul_by_const.restype = POINTER(CMatrix)

c_lib.mat_transpose.argtypes = [POINTER(CMatrix)]
c_lib.mat_transpose.restype = POINTER(CMatrix)

c_lib.mat_trace.argtypes = [POINTER(CMatrix)]
c_lib.mat_trace.restype = c_float

c_lib.mat_identity.argtypes = [c_size_t]
c_lib.mat_identity.restype = POINTER(CMatrix)

c_lib.mat_solve.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
c_lib.mat_solve.restype = POINTER(CMatrix)

c_lib.mat_inv.argtypes = [POINTER(CMatrix)]
c_lib.mat_inv.restype = POINTER(CMatrix)

c_lib.mat_row_echelon.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
c_lib.mat_row_echelon.restype = POINTER(CMatrix)

c_lib.mat_rref.argtypes = [POINTER(CMatrix), POINTER(CMatrix)]
c_lib.mat_rref.restype = POINTER(CMatrix)

c_lib.mat_det.argtypes = [POINTER(CMatrix), POINTER(c_float)]
c_lib.mat_det.restype = c_int

c_lib.mat_free.argtypes = [POINTER(CMatrix)]
c_lib.mat_free.restype = None

def mat_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Non conformable matrices")

    cmat_a, arr_a = prepare_cmatrix(a)
    cmat_b, arr_b = prepare_cmatrix(b)

    raw_out = c_lib.mat_add(byref(cmat_a), byref(cmat_b))
    return from_cmatrix(raw_out)

def mat_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("Non conformable matrices")

    cmat_a, arr_a = prepare_cmatrix(a)
    cmat_b, arr_b = prepare_cmatrix(b)

    raw_out = c_lib.mat_sub(byref(cmat_a), byref(cmat_b))
    return from_cmatrix(raw_out)

def mat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[1] != b.shape[0]:
        raise ValueError("Non conformable matrices")

    cmat_a, arr_a = prepare_cmatrix(a)
    cmat_b, arr_b = prepare_cmatrix(b)

    raw_out = c_lib.mat_mul(byref(cmat_a), byref(cmat_b))
    return from_cmatrix(raw_out)

def mat_mul_by_const(mult: float, a: np.ndarray) -> np.ndarray:
    cmat, arr = prepare_cmatrix(a)
    raw_out = c_lib.mat_mul_by_const(mult, byref(cmat))
    return from_cmatrix(raw_out)

def mat_transpose(a: np.ndarray) -> np.ndarray:
    cmat, arr = prepare_cmatrix(a)
    raw_out = c_lib.mat_transpose(byref(cmat))
    return from_cmatrix(raw_out)

def mat_trace(a: np.ndarray) -> float:
    cmat, arr = prepare_cmatrix(a)
    return c_lib.mat_trace(byref(cmat))

def mat_identity(size: int) -> np.ndarray:
    raw_out = c_lib.mat_identity(size)
    return from_cmatrix(raw_out)

def mat_solve(a: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if a.shape[0] != rhs.shape[0]:
        raise ValueError("Both matrices must have the same number of rows")
    
    rhs_copy = np.array(rhs, dtype = np.float32, copy = True)
    return call_with_rhs(c_lib.mat_solve, a, rhs_copy)

def mat_inv(a: np.ndarray) -> np.ndarray:
    if a.shape[0] != a.shape[1]:
        raise ValueError("Expected a square matrix")
    
    cmat, arr = prepare_cmatrix(a)
    return from_cmatrix(c_lib.mat_inv(byref(cmat)))

def mat_row_echelon(a: np.ndarray, rhs: np.ndarray | None = None) -> np.ndarray:
    return call_with_rhs(c_lib.mat_row_echelon, a, rhs)

def mat_rref(a: np.ndarray, rhs: np.ndarray | None = None) -> np.ndarray:
    return call_with_rhs(c_lib.mat_rref, a, rhs)

def mat_det(a: np.ndarray) -> float:
    if a.shape[0] != a.shape[1]:
        raise ValueError("Expected a square matrix")
    
    cmat, arr = prepare_cmatrix(a)
    result = c_float()
    status = c_lib.mat_det(byref(cmat), byref(result))
    if status != 0:
        raise RuntimeError("Determinant computation failed")

    return result.value

def prepare_cmatrix(a: np.ndarray):
    if a.ndim != 2:
        raise ValueError("Expected 2D array")

    a = np.ascontiguousarray(a, dtype = np.float32)

    cm = CMatrix(
        rows = a.shape[0],
        cols = a.shape[1],
        data = a.ctypes.data_as(POINTER(c_float))
    )

    return cm, a

def from_cmatrix(raw_out) -> np.ndarray:
    if not raw_out:
        raise RuntimeError("C matrix operation failed")

    try:
        out = raw_out.contents

        result = np.ctypeslib.as_array(
            out.data,
            shape=(out.rows, out.cols)
        ).copy()

        return result

    finally:
        c_lib.mat_free(raw_out)

def call_with_rhs(func, a: np.ndarray, rhs: np.ndarray | None = None) -> np.ndarray:
    cmat_a, arr_a = prepare_cmatrix(a)

    if rhs is not None:
        if a.shape[0] != rhs.shape[0]:
            raise ValueError("Both matrices must have the same number of rows")
        
        cmat_rhs, arr_rhs = prepare_cmatrix(rhs)
        rhs_ptr = byref(cmat_rhs)
    else:
        rhs_ptr = None

    raw_out = func(byref(cmat_a), rhs_ptr)

    return from_cmatrix(raw_out)
