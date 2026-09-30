"""Shared numeric single-QP assembly and solve for affine MPC approximations."""
import time
from types import SimpleNamespace

import casadi as ca
import numpy as np
import osqp
from scipy import sparse, linalg

from .qp_condensing import project_qp


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
    return project_qp(qp, basis, offset, equality)


class AffineQPMixin:
    """Requires a quadratic OCP, model-specific warm start and solution retraction."""

    qp_default_backend = "osqp"
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
        self.qp_backend = solver_args.get("backend", self.qp_default_backend)
        if self.qp_backend not in ("osqp", "qpoases", "hpipm"):
            raise ValueError(f"Unknown QP backend: {self.qp_backend}")
        if self.qp_backend == "qpoases" and not ca.has_conic("qpoases"):
            raise RuntimeError("CasADi qpOASES plugin is unavailable; install a CasADi build with qpoases")
        condensed = solver_args.get("condensed", type(self).qp_condense_equalities)
        if not isinstance(condensed, bool):
            raise ValueError("QP condensed must be True or False")
        self.qp_condense_equalities = condensed
        self.hpipm_prob = None
        if self.qp_backend == "hpipm":
            from .hpipm_backend import HPIPMAdapter, HPIPMOCPAdapter
            hpipm_args = dict(mode=solver_args.get("hpipm_mode", "robust"),
                              options=solver_args.get("hpipm_opts", {}),
                              root=solver_args.get("hpipm_root"))
            if not condensed and hasattr(self, "_qp_dynamics_rows"):
                self.hpipm_prob = HPIPMOCPAdapter(self.ndx_opt, self.nu_opt[0],
                                                self._qp_dynamics_rows, **hpipm_args)
                self.qp_hpipm_form = "ocp"
            else:
                self.hpipm_prob = HPIPMAdapter(**hpipm_args)
                self.qp_hpipm_form = "dense"
        self.regularization = float(solver_args.get("regularization", 1e-8))
        self.max_qp_violation = float(solver_args.get("max_qp_violation", 1e-3))
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("QP regularization must be finite and nonnegative")
        if not np.isfinite(self.max_qp_violation) or self.max_qp_violation <= 0:
            raise ValueError("max_qp_violation must be finite and positive")
        self.osqp_opts = dict(self.qp_default_options)
        self.osqp_opts.update(solver_args.get("opts", {}))
        self.qpoases_opts = {"printLevel": "none", "error_on_fail": False}
        self.qpoases_opts.update(solver_args.get("qpoases_opts", {}))
        z, p = self.opti.x, self.opti.p
        # Opti.x / Opti.p rebuild symbolic stacks; the graph is fixed after setup.
        self._qp_z, self._qp_p = z, p
        self._qp_weights = ca.vertcat(self.Q_diag, self.R_diag)
        self._qp_integrate = self.dyn.state_integrate()
        g, l, u = self.opti.g, self.opti.lbg, self.opti.ubg
        hessian, gradient = ca.hessian(self.opti.f, z)
        self.qp_data = ca.Function(f"{self.qp_model_name}_data", [z, p],
                                   [hessian, gradient, ca.jacobian(g, z), g, l, u])
        self.g_data = ca.Function(f"{self.qp_model_name}_constraints", [z, p], [g, l, u])
        self.qp_compile_time = 0.
        self.qp_compiled_library = None
        self.qp_profile = bool(solver_args.get("profile", False))
        if solver_args.get("compile_data", False):
            from .qp_codegen import compile_functions
            compile_start = time.perf_counter()
            (self.qp_data, self.g_data), self.qp_compiled_library = compile_functions(
                [self.qp_data, self.g_data], solver_args.get("codegen_cache"))
            self.qp_compile_time = time.perf_counter() - compile_start
        self._qp_csc_layouts = []
        for output in (0, 2):
            pattern = self.qp_data.sparsity_out(output)
            indptr, indices = pattern.get_ccs()
            self._qp_csc_layouts.append((np.asarray(indices, dtype=np.int32),
                                         np.asarray(indptr, dtype=np.int32), pattern.shape))
        self.osqp_prob = None
        self.qpoases_prob = None
        self._qpoases_shape = None
        self.qp_solve_count = 0

    def build_qp(self):
        """Return numeric P, q, A, l, u for the trajectory correction dz."""
        weights = np.asarray(self.opti.value(self._qp_weights)).ravel()
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Convex QP requires finite, nonnegative Q/R weights")
        reference = np.asarray(self.opti.value(self._qp_z, self.opti.initial())).ravel()
        params = np.asarray(self.opti.value(self._qp_p)).ravel()
        H, gradient, jacobian, g, l, u = self.qp_data(reference, params)
        # This objective is quadratic in the initial-state chart. Its exact
        # cost Hessian is PSD; do not use the nonlinear Lagrangian Hessian.
        numeric = []
        for matrix, (indices, indptr, shape) in zip((H, jacobian), self._qp_csc_layouts):
            numeric.append(sparse.csc_matrix((np.asarray(matrix.nonzeros()), indices.copy(), indptr.copy()),
                                            shape=shape))
        P = sparse.triu(numeric[0] + self.regularization * sparse.eye(reference.size), format="csc")
        qp = dict(P=P, q=np.asarray(gradient).ravel(), A=numeric[1],
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

    def condense_qp(self, qp):
        self.qp_condensing_method = "global_qr"
        return condense_equalities(qp)

    def _solve_osqp(self, solver_qp):
        setup_start = time.perf_counter()
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
        self.qp_setup_time = time.perf_counter() - setup_start
        solver_start = time.perf_counter()
        result = self.osqp_prob.solve()  # Exactly one QP solve per control step.
        self.qp_solver_time = time.perf_counter() - solver_start
        return result.x, result.info.status_val in (1, 2), result.info

    def _solve_qpoases(self, qp):
        setup_start = time.perf_counter()
        n, m = qp["q"].size, qp["A"].shape[0]
        shape = (n, m)
        if self.qpoases_prob is None or shape != self._qpoases_shape:
            self.qpoases_prob = ca.conic(
                f"{self.qp_model_name}_qpoases", "qpoases",
                {"h": ca.Sparsity.dense(n, n), "a": ca.Sparsity.dense(m, n)},
                self.qpoases_opts)
            self._qpoases_shape = shape
        # OSQP stores only the upper triangle. CasADi's conic interface needs
        # the symmetric Hessian, including off-diagonal entries on both sides.
        H = qp["P"] + qp["P"].T - sparse.diags(qp["P"].diagonal())
        arguments = dict(h=ca.DM(H.toarray()), g=ca.DM(qp["q"]),
                         a=ca.DM(qp["A"].toarray()),
                         lba=ca.DM(qp["l"]), uba=ca.DM(qp["u"]))
        self.qp_setup_time = time.perf_counter() - setup_start
        solver_start = time.perf_counter()
        result = self.qpoases_prob(**arguments)  # One active-set QP solve.
        self.qp_solver_time = time.perf_counter() - solver_start
        stats = self.qpoases_prob.stats()
        self.qpoases_stats = stats
        info = SimpleNamespace(status=stats.get("return_status", "unknown"),
                               iter=int(stats.get("iter_count", 0)))
        return np.asarray(result["x"]).ravel(), bool(stats.get("success", False)), info

    def _solve_hpipm(self, qp):
        from .hpipm_backend import canonical_qp
        start = time.perf_counter()
        solver_qp = canonical_qp(qp) if self.qp_hpipm_form == "dense" else qp
        prepare = time.perf_counter() - start
        result = self.hpipm_prob.solve(solver_qp)
        self.qp_setup_time = prepare + result["setup_ms"] / 1000.
        self.qp_solver_time = result["solve_ms"] / 1000.
        self.hpipm_stats = {k: v for k, v in result.items() if k not in ("x", "dual")}
        info = SimpleNamespace(status=result["status"], iter=result["iterations"])
        return result["x"], result["success"], info

    def solve(self, retract_all=True):
        start = time.perf_counter()
        qp = self.build_qp()
        self.qp_assembly_time = time.perf_counter() - start
        self.last_qp = qp
        solver_qp, basis, offset = qp, None, None
        self.qp_condensing_method = "none"
        self.qp_condensing_fallback_reason = None
        condense_start = time.perf_counter()
        if self.qp_condense_equalities:
            solver_qp, basis, offset = self.condense_qp(qp)
        self.qp_condense_time = time.perf_counter() - condense_start
        self.last_solver_qp = solver_qp
        self.qp_build_time = time.perf_counter() - start
        if self.qp_backend == "qpoases":
            solution, success, info = self._solve_qpoases(solver_qp)
        elif self.qp_backend == "hpipm":
            solution, success, info = self._solve_hpipm(solver_qp)
        else:
            solution, success, info = self._solve_osqp(solver_qp)
        self.qp_solve_count += 1
        self.qp_result_info = info
        self.qp_status = info.status
        self.solve_time = time.perf_counter() - start
        if not success or solution is None or not np.all(np.isfinite(solution)):
            raise RuntimeError(f"{self.qp_model_name} failed: {self.qp_status}; no solution applied")
        correction = solution if basis is None else offset + basis @ solution
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
        self.qp_total_time = time.perf_counter() - start
        self.qp_postprocess_time = self.qp_total_time - self.solve_time
        print(f"QP ({self.qp_backend}, condensing={self.qp_condensing_method}): "
              f"{self.qp_status}, {self.solve_time * 1000:.2f} ms, "
              f"affine CV={self.qp_constr_viol:.3g}, nonlinear CV={self.constr_viol:.3g}")
        if self.qp_profile:
            print(f"  assembly={self.qp_assembly_time*1000:.2f}, "
                  f"condense={self.qp_condense_time*1000:.2f}, "
                  f"setup={self.qp_setup_time*1000:.2f}, solve={self.qp_solver_time*1000:.2f}, "
                  f"post={self.qp_postprocess_time*1000:.2f} ms")

    def update_params(self, x_init, t_current):
        self._time = float(t_current)
        self.update_initial_state(x_init)
        self.update_gait_sequence(t_current)
        self.warm_start_variables()

    def _save_prediction(self):
        x_init = np.asarray(self.opti.value(self.x_init)).ravel()
        states = [np.asarray(self._qp_integrate(x_init, dx)).ravel() for dx in self.DX_prev]
        dts = np.array([float(self.opti.value(dt)) for dt in self.dts])
        self._prediction = dict(time=self._time, grid=np.r_[0., np.cumsum(dts)],
                                states=states, inputs=[u.copy() for u in self.U_prev],
                                contacts=np.asarray(self.opti.value(self.contact_schedule)))

