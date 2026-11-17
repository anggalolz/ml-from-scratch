import numpy as np
import pytest

from mlfs.linear_algebra import Matrix

@pytest.fixture
def matrices():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    return A, B

def test_add(matrices):
    A, B = matrices
    expected = np.array([[6, 8], [10, 12]])
    np.testing.assert_allclose((A + B).to_numpy(), expected)

def test_sub(matrices):
    A, B = matrices
    expected = np.array([[-4, -4], [-4, -4]])
    np.testing.assert_allclose((A - B).to_numpy(), expected)

def test_mul(matrices):
    A, B = matrices
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_allclose((A @ B).to_numpy(), expected)

def test_scalar_mul(matrices):
    A, _ = matrices
    expected = np.array([[2, 4], [6, 8]])
    np.testing.assert_allclose((2 * A).to_numpy(), expected)

def test_transpose(matrices):
    A, _ = matrices
    expected = np.array([[1, 3], [2, 4]])
    np.testing.assert_allclose(A.T.to_numpy(), expected)

def test_inverse(matrices):
    A, _ = matrices
    expected = np.linalg.inv(A.to_numpy())
    np.testing.assert_allclose(
        A.inv().to_numpy(), expected, rtol=1e-5, atol=1e-5
    )

def test_identity(matrices):
    A, _ = matrices
    I = Matrix.identity(2)
    np.testing.assert_allclose((A @ I).to_numpy(), A.to_numpy())

def test_trace(matrices):
    A, _ = matrices
    assert A.trace() == pytest.approx(5.0)

def test_determinant(matrices):
    A, _ = matrices
    assert A.det() == pytest.approx(-2.0)

def test_invalid_dimensions():
    A = Matrix([[1, 2]])
    B = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        A + B
