"""Single affine QP with mass-normalized centroidal momentum and velocity inputs.

This retains full robot kinematics. It linearizes the CoM momentum dynamics
and centroidal-map consistency; it is not the fixed-world-origin reduced model.
"""
import numpy as np

from .affine_qp import AffineQPMixin
from .ocp_centroidal_vel import OCPCentroidalVel
from .qp_condensing import condense_stagewise, StageCondensingError


class OCPCentroidalVelQP(AffineQPMixin, OCPCentroidalVel):
    qp_model_name = "centroidal_vel_qp"
    qp_default_backend = "qpoases"
    # Options for the optional OSQP backend only.
    # Momentum consistency and kinematic rows have very different magnitudes.
    # More equilibration passes avoid stalled ADMM iterations at contact switches.
    qp_default_options = {
        "scaling": 50,
        "adaptive_rho_interval": 25,
        # Default infeasibility tolerances can falsely reject feasible late
        # horizon QPs in the condensed coordinates. Tighten certificates only;
        # primal/dual solution tolerances and acceptance bounds are unchanged.
        "eps_prim_inf": 1e-8,
        "eps_dual_inf": 1e-8,
    }
    # Re-equilibrate the current Jacobian and reset ADMM state each control step.
    # Reusing the first QP workspace can stall even when a fresh solve converges.
    qp_rebuild_solver = True
    # Eliminate stage-local equalities, then propagate states through the horizon.
    # A general QR is retained only for degenerate/unsupported stage structures.
    qp_condense_equalities = True

    def __init__(self, robot, nodes, tau_nodes, warm_start, include_base=True):
        if nodes < 2:
            raise ValueError("centroidal_vel_qp requires at least two nodes")
        if not 0 <= tau_nodes <= nodes:
            raise ValueError("tau_nodes must be between zero and nodes")
        super().__init__(robot, nodes, tau_nodes, warm_start, include_base)
        self._qp_dynamics_rows = []
        self._prediction = None
        self._time = 0.0
        self._base_velocity = self.dyn.base_vel_dynamics()
        self._base_acceleration = self.dyn.base_acc_dynamics()

    def condense_qp(self, qp):
        self.qp_condensing_fallback_reason = None
        try:
            reduced = condense_stagewise(qp, self.ndx_opt, self.nu_opt[0],
                                         self._qp_dynamics_rows)
        except StageCondensingError as error:
            # Preserve all compatibility equations at singular configurations.
            # This is an algebraic fallback; the selected QP solver is still called only once.
            self.qp_condensing_fallback_reason = str(error)
            return super().condense_qp(qp)
        self.qp_condensing_method = "stagewise"
        return reduced

    def get_initial_q(self):
        # Unlike RNEA's [q, v], the physical state is [h/m, q].
        return self.x_init[6:]

    def setup_dynamics_constraints(self, i):
        # Use exactly the centroidal NLP's delta-coordinate Euler equations,
        # momentum closure and torque estimates. Only record transition rows
        # for algebraic condensing; no alternative integration model is added.
        row_start = self.opti.ng
        super().setup_dynamics_constraints(i)
        self._qp_dynamics_rows.append(np.arange(row_start, row_start + self.ndx_opt))

    def warm_start_variables(self):
        """Match the original NLP's previous-node-value reuse policy.

        Delta states and velocities are reused at the same node index. The NLP
        resets force guesses from desired support forces/current contacts.
        There is no time shift, chart re-anchoring or base-velocity projection.
        """
        if self.warm_start:
            super().warm_start_variables()

    def retract_stacked_sol(self, sol_x, retract_all=True):
        """Decode first, then reconstruct velocities/accelerations without a terminal input."""
        sol_x = np.asarray(sol_x).ravel()
        x_init = np.asarray(self.opti.value(self.x_init)).ravel()
        stride = self.ndx_opt + self.nu_opt[0]
        self.DX_prev = [sol_x[i * stride:i * stride + self.ndx_opt].copy()
                        for i in range(self.nodes)]
        self.DX_prev.append(sol_x[self.nodes * stride:].copy())
        self.U_prev = [sol_x[i * stride + self.ndx_opt:(i + 1) * stride].copy()
                       for i in range(self.nodes)]
        states = [np.asarray(self._qp_integrate(x_init, dx)).ravel() for dx in self.DX_prev]
        velocities = []
        for x, u in zip(states, self.U_prev):
            if self.include_base:
                velocities.append(u[:self.nv_opt].copy())
            else:
                vj = u[:self.nv_opt]
                vb = np.asarray(self._base_velocity(x[:6], x[6:], vj)).ravel()
                velocities.append(np.r_[vb, vj])
        dts = [float(self.opti.value(dt)) for dt in self.dts]
        for i in range(self.nodes if retract_all else 1):
            h, q = states[i][:6], states[i][6:]
            v = velocities[i]
            f = self.U_prev[i][self.f_idx:]
            if i + 1 < self.nodes:
                aj = (velocities[i + 1][6:] - v[6:]) / dts[i]
            else:
                # No terminal velocity input: use the last available joint-velocity slope.
                aj = (v[6:] - velocities[i - 1][6:]) / dts[i - 1]
            ab = np.asarray(self._base_acceleration(q, v, aj, f)).ravel()
            self.q_sol.append(q.copy())
            self.v_sol.append(v.copy())
            self.a_sol.append(np.r_[ab, aj])
            self.forces_sol.append(f.copy())
        if retract_all:
            self.q_sol.append(states[-1][6:].copy())
