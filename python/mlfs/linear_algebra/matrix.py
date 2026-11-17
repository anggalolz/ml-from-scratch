import numpy as np
from . import _backend

class Matrix:
    def __init__(self, data):
        self._data = np.array(
            data, dtype = np.float32, order = "C", copy = True
        )

        if self._data.ndim != 2:
            raise ValueError("Expected a 2D array")

    @classmethod
    def _from_backend(cls, data):
        """Wrap an array already owned by NumPy without copying."""
        obj = cls.__new__(cls)
        obj._data = data
        return obj

    @classmethod
    def identity(cls, size: int):
        return cls._from_backend(_backend.mat_identity(size))

    @property
    def shape(self):
        return self._data.shape

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    @property
    def T(self):
        return self._from_backend(_backend.mat_transpose(self._data))

    def to_numpy(self):
        return self._data.copy()

    # Arithmetic operations

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented

        return self._from_backend(_backend.mat_add(self._data, other._data))

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented

        return self._from_backend(_backend.mat_sub(self._data, other._data))

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented

        return self._from_backend(_backend.mat_mul(self._data, other._data))

    def __mul__(self, other):
        if not isinstance(other, (int, float, np.number)):
            return NotImplemented

        return self._from_backend(_backend.mat_mul_by_const(other, self._data))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1

    # Linear algebra operations

    def trace(self):
        return _backend.mat_trace(self._data)

    def det(self):
        return _backend.mat_det(self._data)

    def inv(self):
        return self._from_backend(_backend.mat_inv(self._data))

    def solve(self, rhs):
        if not isinstance(rhs, Matrix):
            raise TypeError("Expected a Matrix")

        return self._from_backend(_backend.mat_solve(self._data, rhs._data))

    def row_echelon(self, rhs = None):
        if rhs is not None and not isinstance(rhs, Matrix):
            raise TypeError("Expected a Matrix or None")

        return self._from_backend(
            _backend.mat_row_echelon(
                self._data,
                rhs._data if rhs is not None else None
            )
        )

    def rref(self, rhs = None):
        if rhs is not None and not isinstance(rhs, Matrix):
            raise TypeError("Expected a Matrix or None")

        return self._from_backend(
            _backend.mat_rref(
                self._data,
                rhs._data if rhs is not None else None
            )
        )

    # Representation

    def __repr__(self):
        return f"Matrix({self._data!r})"

    def __str__(self):
        return str(self._data)
