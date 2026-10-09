from .core import jacobi_eigendecomposition, jacobi_eigenvalues

__all__ = ["jacobi_eigendecomposition", "jacobi_eigenvalues"]
"""Public API for the symmetric_matrix_eigendecomposition package.

Exposes two functions:
- jacobi_eigendecomposition: returns eigenvalues and eigenvectors.
- jacobi_eigenvalues: returns only eigenvalues.
"""
