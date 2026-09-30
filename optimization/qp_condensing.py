"""Stagewise equality elimination for a linearized, fixed-initial-state OCP."""
import warnings

import numpy as np
from scipy import linalg, sparse


class StageCondensingError(ValueError):
    """The stagewise parametrization is unsafe; use a general equality reduction."""


def project_qp(qp, basis, offset, equality, H=None, HB=None):
    """Substitute dz = offset + basis @ y, dropping the eliminated equalities."""
    fully_determined = basis.shape[1] == 0
    if fully_determined:
        # Preserve the single solver call even when equalities determine all dz.
        basis = np.zeros((offset.size, 1))
    if H is None:
        H = qp["P"] + qp["P"].T - sparse.diags(qp["P"].diagonal())
    if HB is None or fully_determined:
        HB = H @ basis
    projected_H = basis.T @ HB
    projected_H = .5 * (projected_H + projected_H.T)
    if fully_determined:
        projected_H[0, 0] = 1.
    Ai = qp["A"][~equality]
    shift = Ai @ offset
    reduced = dict(P=sparse.csc_matrix(np.triu(projected_H)),
                   q=basis.T @ (H @ offset + qp["q"]),
                   A=sparse.csc_matrix(Ai @ basis),
                   l=qp["l"][~equality] - shift,
                   u=qp["u"][~equality] - shift)
    return reduced, basis, offset


def condense_stagewise(qp, nx, nu, dynamics_rows):
    """Eliminate local inputs and propagate states, without a horizon-wide QR.

    Layout: [x0, u0, ..., xN-1, uN-1, xN]. The first nx constraint rows
    fix x0. dynamics_rows[k] identifies the nx transition residuals for stage k;
    all other active equality rows must involve only xk and uk. This metadata
    is recorded when the OCP constraints are built, not inferred from Jacobian
    entries that may vanish at a particular linearization point.

    For E*x + F*u = b, a small SVD gives u = K*x + d + Z*y. Substitution
    into the transition determines x_next. Local rank loss or an unsupported
    constraint pattern raises StageCondensingError: callers must retain the
    compatibility equations, e.g. by using general equality elimination.
    """
    N = len(dynamics_rows)
    stride = nx + nu
    n = N * stride + nx
    if qp["q"].size != n:
        raise StageCondensingError("Unexpected stage variable layout")
    A = qp["A"].tocsr(copy=True)
    A.eliminate_zeros()
    lower, upper = qp["l"], qp["u"]
    equality = np.isfinite(lower) & (lower == upper)
    initial = np.arange(nx)
    dynamics_rows = [np.asarray(rows, dtype=int) for rows in dynamics_rows]
    prescribed = np.concatenate([initial, *dynamics_rows])
    if (any(len(rows) != nx for rows in dynamics_rows)
            or len(np.unique(prescribed)) != len(prescribed)
            or np.any(prescribed < 0) or np.any(prescribed >= A.shape[0])
            or not np.all(equality[prescribed])):
        raise StageCondensingError("Invalid initial-state or transition rows")
    if A[initial, nx:].nnz:
        raise StageCondensingError("Initial-state equations depend on future variables")
    remaining = equality.copy()
    remaining[prescribed] = False
    # CSR indices are sorted, so the first/last nonzero column suffice to
    # validate stage locality. Vectorize this instead of visiting every row.
    A.sort_indices()
    nonempty = np.diff(A.indptr) != 0
    if np.any(np.abs(lower[remaining & ~nonempty]) > 1e-7):
        raise ValueError("Inconsistent affine equality constraints; no solution applied")
    local_rows = np.flatnonzero(remaining & nonempty)
    first_stage = A.indices[A.indptr[local_rows]] // stride
    last_stage = A.indices[A.indptr[local_rows+1]-1] // stride
    if np.any(first_stage >= N) or np.any(first_stage != last_stage):
        raise StageCondensingError("Equality couples stages or constrains terminal state")
    local = [local_rows[first_stage == k] for k in range(N)]

    # Materialize once: repeated CSR fancy slicing dominates these small stage
    # operations. Keep CSR for sparse whole-horizon products and validation.
    dense_A = A.toarray()
    # Allocate an upper bound; only the accumulated free columns are multiplied.
    basis = np.zeros((n, N * nu))
    offset = np.zeros(n)
    cursor = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", linalg.LinAlgWarning)
        try:
            offset[:nx] = linalg.solve(dense_A[initial, :nx], lower[initial])
            for k, rows in enumerate(dynamics_rows):
                start = k * stride
                xs = slice(start, start + nx)
                us = slice(start + nx, start + stride)
                ns = slice(start + stride, start + stride + nx)
                block = dense_A[local[k], start:start + stride]
                E, F = block[:, :nx], block[:, nx:]
                U, singular, Vh = linalg.svd(F, full_matrices=True)
                tolerance = (np.finfo(float).eps * max(F.shape)
                             * max(1., singular.max(initial=0.)))
                rank = int(np.count_nonzero(singular > tolerance))
                if rank != F.shape[0]:
                    raise StageCondensingError(f"Stage {k} input equality block loses row rank")
                inverse = (Vh[:rank].T / singular[:rank]) @ U[:, :rank].T
                K = -inverse @ E
                Z = Vh[rank:].T
                basis[us, :cursor] = K @ basis[xs, :cursor]
                basis[us, cursor:cursor + Z.shape[1]] = Z
                offset[us] = inverse @ lower[local[k]] + K @ offset[xs]
                cursor += Z.shape[1]

                if np.any(dense_A[rows, :start]) or np.any(dense_A[rows, ns.stop:]):
                    raise StageCondensingError(f"Stage {k} transition has nonlocal dependencies")
                transition = dense_A[rows, start:ns.stop]
                C = transition[:, :nx]
                B = transition[:, nx:stride]
                D = transition[:, stride:]
                rhs = np.column_stack((lower[rows] - C @ offset[xs] - B @ offset[us],
                                       -C @ basis[xs, :cursor] - B @ basis[us, :cursor]))
                solution = linalg.solve(D, rhs)
                offset[ns] = solution[:, 0]
                basis[ns, :cursor] = solution[:, 1:]
        except (linalg.LinAlgError, linalg.LinAlgWarning) as error:
            raise StageCondensingError("Singular or ill-conditioned stage elimination") from error
    basis = np.ascontiguousarray(basis[:, :cursor])
    # Normalize curvature in the free coordinates. Stage elimination mixes
    # velocity and force units; this diagonal change of coordinates preserves
    # causal zeros and the original objective, unlike adding regularization here.
    H = qp["P"] + qp["P"].T - sparse.diags(qp["P"].diagonal())
    HB = H @ basis
    curvature = np.sum(basis * HB, axis=0)
    scale = np.ones(cursor)
    positive = curvature > 0.
    scale[positive] = 1. / np.maximum(np.sqrt(curvature[positive]), 1e-6)
    basis *= scale
    HB *= scale
    # Verify the affine parametrization before dropping any original equation.
    Ae = A[equality]
    error = max(np.max(np.abs(Ae @ basis), initial=0.),
                np.max(np.abs(Ae @ offset - lower[equality]), initial=0.))
    if (not np.all(np.isfinite(basis)) or not np.all(np.isfinite(offset))
            or not np.isfinite(error) or error > 1e-7):
        raise StageCondensingError("Stagewise reconstruction exceeds equality tolerance")
    return project_qp(qp, basis, offset, equality, H=H, HB=HB)
