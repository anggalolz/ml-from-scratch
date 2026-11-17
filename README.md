# Machine Learning from Scratch

This is a personal project to explore the mathematics and implementation of machine learning algorithms from scratch.

The core algorithms are implemented in **C**, with **Python** providing a high-level interface for data manipulation, experimentation, and visualisation.

## Development Rules

- **From scratch:** Core mathematical operations and machine learning algorithms are implemented manually.
- **Python interoperability:** NumPy and pandas may be used for data representation and manipulation.
- **Verification:** NumPy, SciPy, and scikit-learn may be used as reference implementations for testing.
- **AI usage:** Strictly used only for writing automated testing and C-Python interoperability.

## Project Structure

```text
ml-from-scratch/
├── include/ml/             # Public C headers
├── src/                    # C implementations
│   ├── linalg/
|   └── models/
├── python/mlfs/            # Python package and C bindings
│   └── linear_algebra/
├── tests/                  # C and Python tests
├── examples/               # Usage examples
├── environment.yml         # Conda environment
└── pyproject.toml          # Python package configuration
```

## Getting Started

### 1. Set up the Python environment

Install [Conda](https://docs.conda.io/) and create the development environment:

```bash
conda env create -f environment.yml
conda activate mlfs
```

Install the Python package in editable mode:

```bash
python -m pip install -e .
```

### 2. Compile the C library

A C compiler such as GCC or Clang is required.

On Linux or macOS, compile the shared library using:

```bash
mkdir -p build/lib

gcc -O3 -Wall -Wextra -fPIC -shared \
    -Iinclude src/linalg/*.c \
    -o build/lib/matrix.so
```

The compiled library is loaded by the Python backend through `ctypes`.

### 3. Run tests

```bash
python -m pytest tests/python/ -v
```

## Example Usage

```python
from mlfs.linear_algebra import Matrix

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print(A + B)       # Addition
print(A @ B)       # Matrix multiplication
print(A.T)         # Transpose
print(A.inv())     # Inverse
print(A.det())     # Determinant
```

All core matrix computations are performed by the C implementation.

## License

See [LICENSE](LICENSE) for licensing information.