"""One affine whole-body horizon QP per MPC update (no SQP loop).

CasADi evaluates the nonlinear model and its first derivatives outside OSQP.
The QP solves for a correction to a shifted trajectory. Nonlinear feasibility
is reported separately; feasibility of the affine QP does not imply it.
"""
import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin

from .ocp_whole_body_rnea import OCPWholeBodyRNEA
from .affine_qp import AffineQPMixin


class OCPWholeBodyRNEAQP(AffineQPMixin, OCPWholeBodyRNEA):
    qp_model_name = "whole_body_rnea_qp"

    def __init__(self, robot, nodes, tau_nodes, warm_start, include_acc=True):
        if not include_acc:
            raise ValueError("whole_body_rnea_qp requires include_acc=True")
        if nodes < 2:
            raise ValueError("whole_body_rnea_qp requires at least two nodes")
        # Keep acceleration, force and torque variables at EVERY input node.
        super().__init__(robot, nodes, nodes, warm_start, include_acc=True)
        self._prediction = None
        self._time = 0.0
        self._rnea = self.dyn.rnea_dynamics()
        q = ca.SX.sym("q", self.nq)
        dq = ca.SX.sym("dq", self.nv)
        q1 = ca.SX.sym("q1", self.nq)
        self._integrate_q = ca.Function("qp_integrate_q", [q, dq],
                                         [cpin.integrate(self.dyn.model, q, dq)])
        self._difference_q = ca.Function("qp_difference_q", [q, q1],
                                          [cpin.difference(self.dyn.model, q, q1)])

    def setup_dynamics_constraints(self, i):
        q, v = self.get_q(i), self.get_v(i)
        a, f, tau = self.get_a(i), self.get_forces(i), self.get_tau(i)
        dt = self.dts[i]
        # Evaluate and differentiate manifold integration rather than adding
        # angular coordinates in a chart anchored at the initial pose.
        q_next = self._integrate_q(q, dt * v)
        self.opti.subject_to(self._difference_q(q_next, self.get_q(i + 1)) == 0)
        self.opti.subject_to(self.get_v(i + 1) == v + dt * a)
        self.opti.subject_to(self._rnea(q, v, a, f) == ca.vertcat(ca.DM.zeros(6), tau))
        self.opti.subject_to(self.opti.bounded(-self.robot.joint_torque_max,
                                             tau, self.robot.joint_torque_max))

    def setup_constraints(self, mu=0.9):
        super().setup_constraints(mu)
        # The common constraints omit the terminal node. Bound its joints too.
        self.opti.subject_to(self.opti.bounded(self.robot.joint_pos_min,
                                             self.get_q(self.nodes)[7:],
                                             self.robot.joint_pos_max))
        self.opti.subject_to(self.opti.bounded(-self.robot.joint_vel_max,
                                             self.get_v(self.nodes)[6:],
                                             self.robot.joint_vel_max))

    def warm_start_variables(self):
        """Time-shift physical states, then re-anchor in the measured-state chart."""
        x_init = np.asarray(self.opti.value(self.x_init)).ravel()
        q0, v0 = x_init[:self.nq], x_init[self.nq:]
        dts = np.array([float(self.opti.value(dt)) for dt in self.dts])
        grid = np.r_[0., np.cumsum(dts)]
        schedule = np.asarray(self.opti.value(self.contact_schedule))
        previous = self._prediction if self.warm_start else None
        f_des = np.asarray(self.opti.value(self.f_des)).ravel()
        for k, t in enumerate(grid):
            if previous is None:
                q = pin.integrate(self.model, q0, t * v0)
                v = v0.copy()
                u = np.r_[np.zeros(self.nv), f_des, np.zeros(self.nj)]
                old_contact = None
            else:
                old_t = np.clip(self._time - previous["time"] + t, 0., previous["grid"][-1])
                j = int(np.clip(np.searchsorted(previous["grid"], old_t, side="right") - 1,
                                0, self.nodes - 1))
                alpha = (old_t - previous["grid"][j]) / (previous["grid"][j + 1] - previous["grid"][j])
                x0, x1 = previous["states"][j:j + 2]
                q = pin.integrate(self.model, x0[:self.nq],
                                  alpha * pin.difference(self.model, x0[:self.nq], x1[:self.nq]))
                v = (1. - alpha) * x0[self.nq:] + alpha * x1[self.nq:]
                # Inputs are held piecewise constant, not interpolated across a contact switch.
                u = previous["inputs"][j].copy()
                old_contact = previous["contacts"][:, j]
            if k == 0:
                q, v = q0, v0
            dx = np.r_[pin.difference(self.model, q0, q), v - v0]
            self.opti.set_initial(self.DX_opt[k], dx)
            if k == self.nodes:
                continue
            for foot in range(self.n_feet):
                sl = slice(self.f_idx + 3 * foot, self.f_idx + 3 * foot + 3)
                if schedule[foot, k] == 0:
                    u[sl] = 0.
                elif old_contact is None or old_contact[foot] == 0:
                    u[sl] = f_des[3 * foot:3 * foot + 3]
            if self.arm_ee_frame:
                u[self.f_idx + 3 * self.n_feet:self.tau_idx] = np.asarray(self.opti.value(self.arm_force_des)).ravel()
            if previous is None:
                tau = np.asarray(self._rnea(q, v, u[:self.na_opt], u[self.f_idx:self.tau_idx])).ravel()[6:]
                u[self.tau_idx:] = np.clip(tau, -self.robot.joint_torque_max, self.robot.joint_torque_max)
            self.opti.set_initial(self.U_opt[k], u)
