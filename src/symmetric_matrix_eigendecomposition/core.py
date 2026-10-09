from __future__ import annotations

import math
from typing import List, Sequence, Tuple


Matrix = Sequence[Sequence[float]]


def _validate_square(matrix: Matrix) -> int:
    """Validate that *matrix* is a non-empty square list-of-lists. Returns n."""
    if not isinstance(matrix, (list, tuple)):
        raise TypeError("matrix must be a list or tuple of lists")
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must not be empty")
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise TypeError(f"row {i} must be a list or tuple")
        if len(row) != n:
            raise ValueError(f"matrix is not square: row {i} has length {len(row)}, expected {n}")
    return n


def _is_symmetric(matrix: Matrix, n: int, tol: float) -> None:
    """Raise ValueError if the matrix is not symmetric within *tol*.

    Symmetry is a hard precondition for the Jacobi algorithm: it relies on the
    guarantee that off-diagonal mass is driven to zero by plane rotations.
    Allowing asymmetry silently would return nonsense eigenpairs.
    """
    for i in range(n):
        row_i = matrix[i]
        for j in range(i + 1, n):
            if abs(row_i[j] - matrix[j][i]) > tol:
                raise ValueError(
                    f"matrix is not symmetric at ({i},{j}): "
                    f"{row_i[j]} vs {matrix[j][i]} (tol={tol})"
                )


def jacobi_eigendecomposition(
    matrix: Matrix,
    *,
    tol: float = 1e-12,
    max_sweeps: int = 100,
) -> Tuple[List[float], List[List[float]]]:
    """Diagonalize a real symmetric matrix using the cyclic Jacobi algorithm.

    Args:
        matrix: An n x n real symmetric matrix as a list (or tuple) of lists of floats.
        tol: Convergence threshold on the largest off-diagonal element.
        max_sweeps: Maximum number of full cyclic sweeps over the upper triangle.

    Returns:
        A pair (eigenvalues, eigenvectors):
        - eigenvalues: list of n floats, ordered by descending magnitude.
        - eigenvectors: list of n lists of n floats. eigenvectors[k] is the column
          vector corresponding to eigenvalues[k], so A @ eigenvectors[k] is
          approximately eigenvalues[k] * eigenvectors[k].

    Raises:
        TypeError: if the input is not a list/tuple of lists/tuples.
        ValueError: if the matrix is empty, non-square, or not symmetric within tol.

    The Jacobi method is chosen because it is unconditionally stable for symmetric
    inputs and uses only elementary arithmetic; no QR factorization or Hessenberg
    reduction is needed. The trade-off is O(n^3) per sweep with several sweeps,
    which is slower than divide-and-conquer methods for large n. It is intended for
    moderate-size dense matrices where simplicity and robustness matter.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1")

    n = _validate_square(matrix)
    _is_symmetric(matrix, n, tol)

    # Copy the working matrix into a mutable, dense list-of-lists. We never read
    # the lower triangle again, so we copy only the upper triangle plus diagonal.
    a: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        a[i][i] = float(matrix[i][i])
        for j in range(i + 1, n):
            v = float(matrix[i][j])
            a[i][j] = v
            a[j][i] = v

    # V accumulates the rotations; starts as the identity.
    v: List[List[float]] = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        off = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        if off <= tol * tol:
            break

        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= 1e-300:
                    continue
                app = a[p][p]
                aqq = a[q][q]
                diff = aqq - app
                if abs(apq) < 1e-300:
                    continue
                # If the diagonal entries differ enough, use the stable formula
                # for the rotation angle; otherwise theta is pi/4.
                if abs(diff) < 1e-300:
                    theta = math.pi / 4.0
                else:
                    t = diff / (2.0 * apq)
                    # Root with larger absolute value keeps |tan| <= 1, ensuring
                    # the rotation is the smaller-angle one and numerically tame.
                    disc = math.sqrt(1.0 + t * t)
                    u = t + math.copysign(disc, t) if t != 0.0 else disc
                    theta = math.atan(1.0 / u)

                c = math.cos(theta)
                s = math.sin(theta)

                # Apply rotation J^T A J in place: update rows/cols p and q.
                # Precompute new diagonal entries.
                new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq
                new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq
                a[p][p] = new_app
                a[q][q] = new_aqq
                a[p][q] = 0.0
                a[q][p] = 0.0

                for i in range(n):
                    if i == p or i == q:
                        continue
                    aip = a[i][p]
                    aiq = a[i][q]
                    a[i][p] = c * aip - s * aiq
                    a[p][i] = a[i][p]
                    a[i][q] = s * aip + c * aiq
                    a[q][i] = a[i][q]

                # Accumulate eigenvectors: V = V * J(p, q, theta).
                for i in range(n):
                    vip = v[i][p]
                    viq = v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq

    # Extract diagonal eigenvalues and pair them with eigenvector columns.
    pairs: List[Tuple[float, int]] = [(a[i][i], i) for i in range(n)]
    # Sort by descending magnitude. Ties are broken by index for determinism.
    pairs.sort(key=lambda kv: (-abs(kv[0]), kv[1]))

    eigenvalues: List[float] = [a[i][i] for _, i in pairs]
    eigenvectors: List[List[float]] = [[v[row][i] for _, i in pairs] for row in range(n)]

    return eigenvalues, eigenvectors


def jacobi_eigenvalues(
    matrix: Matrix,
    *,
    tol: float = 1e-12,
    max_sweeps: int = 100,
) -> List[float]:
    """Return only the eigenvalues of a real symmetric matrix, descending by magnitude.

    Thin wrapper around :func:`jacobi_eigendecomposition` for callers that do not
    need the eigenvectors. Provided as a separate export so the intent is explicit
    in calling code.
    """
    eigenvalues, _ = jacobi_eigendecomposition(matrix, tol=tol, max_sweeps=max_sweeps)
    return eigenvalues
