"""Single affine QP with mass-normalized centroidal momentum and velocity inputs.

This retains full robot kinematics. It linearizes the CoM momentum dynamics
and centroidal-map consistency; it is not the fixed-world-origin reduced model.
"""
import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from .affine_qp import AffineQPMixin
from .ocp_centroidal_vel import OCPCentroidalVel


class OCPCentroidalVelQP(AffineQPMixin, OCPCentroidalVel):
    qp_model_name = "centroidal_vel_qp"
    # Momentum consistency and kinematic rows have very different magnitudes.
    # More equilibration passes avoid stalled ADMM iterations at contact switches.
    qp_default_options = {"scaling": 50}
    # Re-equilibrate the current Jacobian and reset ADMM state each control step.
    # Reusing the first QP workspace can stall even when a fresh solve converges.
    qp_rebuild_solver = True
    # Enforce equalities algebraically and optimize in their nullspace. This
    # avoids stalled dual convergence of the much larger equality-constrained QP.
    qp_condense_equalities = True

    def __init__(self, robot, nodes, tau_nodes, warm_start, include_base=True):
        if nodes < 2:
            raise ValueError("centroidal_vel_qp requires at least two nodes")
        if not 0 <= tau_nodes <= nodes:
            raise ValueError("tau_nodes must be between zero and nodes")
        super().__init__(robot, nodes, tau_nodes, warm_start, include_base)
        self._prediction = None
        self._time = 0.0
        q = ca.SX.sym("q", self.nq)
        dq = ca.SX.sym("dq", self.nv)
        q1 = ca.SX.sym("q1", self.nq)
        self._integrate_q = ca.Function("centroidal_qp_integrate", [q, dq],
                                       [cpin.integrate(self.dyn.model, q, dq)])
        self._difference_q = ca.Function("centroidal_qp_difference", [q, q1],
                                        [cpin.difference(self.dyn.model, q, q1)])
        self._com_dynamics = self.dyn.com_dynamics()
        self._momentum_gap = self.dyn.dynamics_gaps()
        self._base_velocity = self.dyn.base_vel_dynamics()
        self._base_acceleration = self.dyn.base_acc_dynamics()
        self._torque_estimate = self.dyn.tau_estimate()

    def get_initial_q(self):
        # Unlike RNEA's [q, v], the physical state is [h/m, q].
        return self.x_init[6:]

    def setup_dynamics_constraints(self, i):
        h, q = self.get_h(i), self.get_q(i)
        v, f = self.get_v(i), self.get_forces(i)
        dt = self.dts[i]
        self.opti.subject_to(self.get_h(i + 1) == h + dt * self._com_dynamics(q, f))
        q_next = self._integrate_q(q, dt * v)
        self.opti.subject_to(self._difference_q(q_next, self.get_q(i + 1)) == 0)
        if self.include_base:
            self.opti.subject_to(self._momentum_gap(h, q, v) == 0)
        if i < self.tau_nodes:
            # Same quasi-static torque estimate as centroidal_vel, NOT full RNEA limits.
            self.opti.subject_to(self.opti.bounded(-self.robot.joint_torque_max,
                                                 self._torque_estimate(q, f),
                                                 self.robot.joint_torque_max))

    def setup_constraints(self, mu=0.9):
        super().setup_constraints(mu)
        self.opti.subject_to(self.opti.bounded(self.robot.joint_pos_min,
                                             self.get_q(self.nodes)[7:],
                                             self.robot.joint_pos_max))
        # Velocities are inputs: bounds exist at 0..N-1, not at the terminal state.

    def warm_start_variables(self):
        x_init = np.asarray(self.opti.value(self.x_init)).ravel()
        h0, q0 = x_init[:6], x_init[6:]
        dts = np.array([float(self.opti.value(dt)) for dt in self.dts])
        grid = np.r_[0., np.cumsum(dts)]
        schedule = np.asarray(self.opti.value(self.contact_schedule))
        previous = self._prediction if self.warm_start else None
        f_des = np.asarray(self.opti.value(self.f_des)).ravel()
        v0 = np.r_[np.asarray(self._base_velocity(h0, q0, np.zeros(self.nj))).ravel(),
                    np.zeros(self.nj)]
        for k, t in enumerate(grid):
            if previous is None:
                h = h0.copy()
                q = pin.integrate(self.model, q0, t * v0)
                u = np.r_[np.zeros(self.nv_opt), f_des]
                old_contact = None
            else:
                old_t = np.clip(self._time - previous["time"] + t, 0., previous["grid"][-1])
                j = int(np.clip(np.searchsorted(previous["grid"], old_t, side="right") - 1,
                                0, self.nodes - 1))
                alpha = (old_t - previous["grid"][j]) / (previous["grid"][j + 1] - previous["grid"][j])
                xa, xb = previous["states"][j:j + 2]
                h = (1. - alpha) * xa[:6] + alpha * xb[:6]
                q = pin.integrate(self.model, xa[6:], alpha * pin.difference(self.model, xa[6:], xb[6:]))
                u = previous["inputs"][j].copy()
                old_contact = previous["contacts"][:, j]
            if k == 0:
                h, q = h0, q0
            self.opti.set_initial(self.DX_opt[k], np.r_[h - h0, pin.difference(self.model, q0, q)])
            if k == self.nodes:
                continue
            if self.include_base:
                # Project the reference velocity onto A(q)v = m*h without a solve.
                u[:6] = np.asarray(self._base_velocity(h, q, u[6:self.nv_opt])).ravel()
            for foot in range(self.n_feet):
                sl = slice(self.f_idx + 3 * foot, self.f_idx + 3 * foot + 3)
                if schedule[foot, k] == 0:
                    u[sl] = 0.
                elif old_contact is None or old_contact[foot] == 0:
                    u[sl] = f_des[3 * foot:3 * foot + 3]
            if self.arm_ee_frame:
                u[self.f_idx + 3 * self.n_feet:] = np.asarray(self.opti.value(self.arm_force_des)).ravel()
            self.opti.set_initial(self.U_opt[k], u)

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
        states = [np.asarray(self.dyn.state_integrate()(x_init, dx)).ravel() for dx in self.DX_prev]
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
