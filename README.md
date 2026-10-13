# symmetric_matrix_eigendecomposition

Diagonalizes real symmetric matrices using the cyclic Jacobi eigenvalue algorithm. Standard library only — no numpy, no scipy, no network access required.

## Usage

```python
from symmetric_matrix_eigendecomposition import jacobi_eigendecomposition

A = [[4.0, 1.0, 0.0],
     [1.0, 3.0, 1.0],
     [0.0, 1.0, 2.0]]

eigenvalues, eigenvectors = jacobi_eigendecomposition(A)
# eigenvalues[k] is a float, ordered by descending magnitude.
# eigenvectors[row][k] is the k-th eigenvector's row-th component,
# so sum(A[row][i] * eigenvectors[i][k] for i in range(n)) approximates
# eigenvalues[k] * eigenvectors[row][k].
```

For eigenvalues only:

```python
from symmetric_matrix_eigendecomposition import jacobi_eigenvalues

A = [[4.0, 1.0, 0.0],
     [1.0, 3.0, 1.0],
     [0.0, 1.0, 2.0]]

vals = jacobi_eigenvalues(A)
```

## Why this exists

The Jacobi method diagonalizes a real symmetric matrix by applying a sequence of plane rotations that zero out the largest off-diagonal entries. It is unconditionally stable for symmetric inputs and uses only elementary arithmetic, which makes it a good fit for environments where numpy cannot be installed — a sealed container, a constrained runtime, or a small embedded tool.

The trade-off is cost: each full sweep is O(n^3) and convergence takes several sweeps, so this is slower than divide-and-conquer solvers for large n. It is intended for moderate-size dense symmetric matrices (say, n up to a few hundred) where robustness and zero dependencies matter more than peak throughput.

## Edge cases and limits

- The matrix must be symmetric within `tol` (default `1e-12`). Asymmetric inputs raise `ValueError` rather than returning silently wrong answers; Jacobi's convergence guarantee depends on symmetry.
- Empty or non-square matrices raise `ValueError`.
- Inputs must be `list`/`tuple` of `list`/`tuple`. Anything else raises `TypeError`.
- Repeated eigenvalues are fine — the eigenvectors returned are a valid basis of the eigenspace, but their specific orientation within a degenerate subspace is arbitrary.
- `max_sweeps` defaults to 100; for well-conditioned symmetric matrices 10 or fewer sweeps usually suffice. If the limit is hit without convergence, the result is returned anyway and will be as close as the last sweep achieved.
- Eigenvalues are returned in descending order of magnitude; eigenvector columns are matched to that order.

## Exported names

- `jacobi_eigendecomposition(matrix, *, tol=1e-12, max_sweeps=100)` -> `(eigenvalues, eigenvectors)`
- `jacobi_eigenvalues(matrix, *, tol=1e-12, max_sweeps=100)` -> `eigenvalues`

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

