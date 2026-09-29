"""Shared numeric single-QP assembly and solve for affine MPC approximations."""
import time

import casadi as ca
import numpy as np
import osqp
from scipy import sparse, linalg


def _csc(matrix):
    """Preserve structural zeros, including entries changed by contact switches."""
    rows, cols = matrix.sparsity().get_triplet()
    return sparse.csc_matrix((np.asarray(matrix.nonzeros()), (rows, cols)),
                             shape=matrix.shape)


def condense_equalities(qp):
    """Eliminate linear equalities with an orthonormal nullspace (same QP).

    The original correction is offset + basis @ y. No optimization is performed
    here; a rank-revealing QR handles zero/redundant equality rows.
    """
    A, lower, upper = qp["A"], qp["l"], qp["u"]
    equality = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    if not np.any(equality):
        return qp, None, np.zeros(A.shape[1])
    Ae, be = A[equality].toarray(), lower[equality]
    Q, R, pivots = linalg.qr(Ae.T, mode="full", pivoting=True)
    diagonal = np.abs(np.diag(R))
    tolerance = np.finfo(float).eps * max(Ae.shape) * max(1., diagonal.max(initial=0.))
    rank = int(np.count_nonzero(diagonal > tolerance))
    offset = np.zeros(A.shape[1])
    if rank:
        offset = Q[:, :rank] @ linalg.solve_triangular(
            R[:rank, :rank].T, be[pivots[:rank]], lower=True)
    basis = Q[:, rank:]
    if np.max(np.abs(Ae @ offset - be), initial=0.) > 1e-7:
        raise ValueError("Inconsistent affine equality constraints; no solution applied")
    if basis.shape[1] == 0:
        # A fully determined primal still uses one trivial QP solver call.
        basis = np.zeros((A.shape[1], 1))
    H = qp["P"] + qp["P"].T - sparse.diags(qp["P"].diagonal())
    projected_H = basis.T @ H @ basis
    projected_H = .5 * (projected_H + projected_H.T)
    if rank == A.shape[1]:
        projected_H[0, 0] = 1.
    Ai = A[~equality]
    reduced = dict(P=sparse.csc_matrix(np.triu(projected_H)),
                   q=basis.T @ (H @ offset + qp["q"]),
                   A=sparse.csc_matrix(Ai @ basis),
                   l=lower[~equality] - Ai @ offset,
                   u=upper[~equality] - Ai @ offset)
    return reduced, basis, offset


class AffineQPMixin:
    """Requires a quadratic OCP, model-specific warm start and solution retraction."""

    qp_default_options = {}
    qp_rebuild_solver = False
    qp_condense_equalities = False

    def setup_friction_constraints(self, force, in_contact, mu):
        # Inscribed pyramid: |fx| + |fy| <= mu*fz (four linear faces).
        self.opti.subject_to(in_contact * force[2] >= 0)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            self.opti.subject_to(in_contact * (sx * force[0] + sy * force[1]
                                              - mu * force[2]) <= 0)

    def init_solver(self, solver, solver_args):
        if solver != "qp":
            raise ValueError(f'{self.qp_model_name} must be used with solver="qp"')
        self.solver = solver
        self.regularization = float(solver_args.get("regularization", 1e-8))
        self.max_qp_violation = float(solver_args.get("max_qp_violation", 1e-3))
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("QP regularization must be finite and nonnegative")
        if not np.isfinite(self.max_qp_violation) or self.max_qp_violation <= 0:
            raise ValueError("max_qp_violation must be finite and positive")
        self.osqp_opts = dict(self.qp_default_options)
        self.osqp_opts.update(solver_args.get("opts", {}))
        z, p = self.opti.x, self.opti.p
        g, l, u = self.opti.g, self.opti.lbg, self.opti.ubg
        hessian, gradient = ca.hessian(self.opti.f, z)
        self.qp_data = ca.Function(f"{self.qp_model_name}_data", [z, p],
                                   [hessian, gradient, ca.jacobian(g, z), g, l, u])
        self.g_data = ca.Function(f"{self.qp_model_name}_constraints", [z, p], [g, l, u])
        self.osqp_prob = None
        self.qp_solve_count = 0

    def build_qp(self):
        """Return numeric P, q, A, l, u for the trajectory correction dz."""
        weights = np.concatenate((np.asarray(self.opti.value(self.Q_diag)).ravel(),
                                  np.asarray(self.opti.value(self.R_diag)).ravel()))
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Convex QP requires finite, nonnegative Q/R weights")
        reference = np.asarray(self.opti.value(self.opti.x, self.opti.initial())).ravel()
        params = np.asarray(self.opti.value(self.opti.p)).ravel()
        H, gradient, jacobian, g, l, u = self.qp_data(reference, params)
        # This objective is quadratic in the initial-state chart. Its exact
        # cost Hessian is PSD; do not use the nonlinear Lagrangian Hessian.
        P = sparse.triu(_csc(H) + self.regularization * sparse.eye(reference.size), format="csc")
        qp = dict(P=P, q=np.asarray(gradient).ravel(), A=_csc(jacobian),
                  l=np.asarray(l - g).ravel(), u=np.asarray(u - g).ravel())
        if not all(np.all(np.isfinite(v)) for v in (P.data, qp["q"], qp["A"].data)):
            raise ValueError("Non-finite QP coefficients")
        if np.any(np.isnan(qp["l"])) or np.any(np.isnan(qp["u"])):
            raise ValueError("NaN QP bounds")
        if np.any(qp["l"] > qp["u"]):
            raise ValueError("Inconsistent QP bounds")
        self.qp_reference, self.qp_params = reference, params
        return qp

    @staticmethod
    def _pattern(matrix):
        return (matrix.shape, matrix.indptr.tobytes(), matrix.indices.tobytes())

    def solve(self, retract_all=True):
        start = time.perf_counter()
        qp = self.build_qp()
        self.last_qp = qp
        solver_qp, basis, offset = qp, None, None
        if self.qp_condense_equalities:
            solver_qp, basis, offset = condense_equalities(qp)
        self.last_solver_qp = solver_qp
        self.qp_build_time = time.perf_counter() - start
        pattern = (self._pattern(solver_qp["P"]), self._pattern(solver_qp["A"]))
        if self.osqp_prob is None or self.qp_rebuild_solver or pattern != self._qp_pattern:
            self.osqp_prob = osqp.OSQP()
            self.osqp_prob.setup(**solver_qp, **self.osqp_opts)
            self._qp_pattern = pattern
        else:
            self.osqp_prob.update(Px=solver_qp["P"].data, q=solver_qp["q"], Ax=solver_qp["A"].data,
                                  l=solver_qp["l"], u=solver_qp["u"])
        # The previous physical solution has already been shifted into the
        # reference. Its old correction is not a valid warm start in this chart.
        self.osqp_prob.warm_start(x=np.zeros(solver_qp["q"].size))
        result = self.osqp_prob.solve()  # Exactly one QP solve per control step.
        self.qp_solve_count += 1
        self.qp_result_info = result.info
        self.qp_status = result.info.status
        self.solve_time = time.perf_counter() - start
        if result.info.status_val not in (1, 2) or result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"{self.qp_model_name} failed: {self.qp_status}; no solution applied")
        correction = result.x if basis is None else offset + basis @ result.x
        self.qp_constr_viol = float(self.constr_viol_norm_inf(qp["A"] @ correction, qp["l"], qp["u"]))
        if self.qp_constr_viol > self.max_qp_violation:
            raise RuntimeError(f"{self.qp_model_name} residual {self.qp_constr_viol:.3g} exceeds "
                               f"{self.max_qp_violation:.3g}; no solution applied")
        sol_z = self.qp_reference + correction
        g, l, u = self.g_data(sol_z, self.qp_params)
        self.constr_viol = float(self.constr_viol_norm_inf(g, l, u))
        self.last_solution = sol_z
        self.retract_stacked_sol(sol_z, retract_all)
        self._save_prediction()
        print(f"QP: {self.qp_status}, {self.solve_time * 1000:.2f} ms, "
              f"affine CV={self.qp_constr_viol:.3g}, nonlinear CV={self.constr_viol:.3g}")

    def update_params(self, x_init, t_current):
        self._time = float(t_current)
        self.update_initial_state(x_init)
        self.update_gait_sequence(t_current)
        self.warm_start_variables()

    def _save_prediction(self):
        x_init = np.asarray(self.opti.value(self.x_init)).ravel()
        states = [np.asarray(self.dyn.state_integrate()(x_init, dx)).ravel() for dx in self.DX_prev]
        dts = np.array([float(self.opti.value(dt)) for dt in self.dts])
        self._prediction = dict(time=self._time, grid=np.r_[0., np.cumsum(dts)],
                                states=states, inputs=[u.copy() for u in self.U_prev],
                                contacts=np.asarray(self.opti.value(self.contact_schedule)))

