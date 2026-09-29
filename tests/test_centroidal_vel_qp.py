"""Centroidal velocity QP regressions, including reduced velocity inputs."""
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np
import osqp
import pinocchio as pin
from scipy import sparse

from args import SOLVER_ARGS
from optimization import make_ocp
from utils.robot import B2_Z1


class CentroidalVelocityQPTests(unittest.TestCase):
    def make_problem(self, include_base=True, gait="stand"):
        robot = B2_Z1(reference_pose="standing_with_arm_up", arm_joints=4)
        robot.set_gait_sequence(gait, .8)
        o = make_ocp("centroidal_vel_qp", {"include_base": include_base},
                     robot=robot, nodes=4, tau_nodes=2, warm_start=True)
        o.set_time_params(.015, .04)
        o.set_swing_params(.07, [.1, -.2])
        o.set_tracking_targets(np.zeros(6), np.zeros(3), np.zeros(3))
        o.update_params(o.x_nom, 0.)
        o.init_solver("qp", SOLVER_ARGS["qp"])
        return o

    def test_state_input_dimensions_convexity_and_jacobian(self):
        for include_base in (True, False):
            with self.subTest(include_base=include_base):
                o = self.make_problem(include_base)
                self.assertEqual(o.nx, 6 + o.nq)
                self.assertEqual(o.ndx_opt, 6 + o.nv)
                self.assertEqual(o.nu_opt[0], (o.nv if include_base else o.nj) + o.nf)
                self.assertEqual(o.tau_nodes, 2)
                qp = o.build_qp()
                self.assertEqual((qp["P"] - sparse.diags(qp["P"].diagonal())).nnz, 0)
                self.assertGreater(qp["P"].diagonal().min(), 0.)
                direction = np.random.default_rng(8).normal(size=o.opti.nx)
                direction /= np.linalg.norm(direction)
                eps = 1e-5
                gp = np.asarray(o.g_data(o.qp_reference + eps * direction, o.qp_params)[0]).ravel()
                gm = np.asarray(o.g_data(o.qp_reference - eps * direction, o.qp_params)[0]).ravel()
                np.testing.assert_allclose((gp - gm) / (2 * eps), qp["A"] @ direction,
                                           atol=2e-6, rtol=2e-5)

    def test_full_retraction_and_mass_normalized_momentum(self):
        for include_base in (True, False):
            with self.subTest(include_base=include_base):
                o = self.make_problem(include_base)
                o.solve(retract_all=True)
                self.assertEqual(len(o.DX_prev), o.nodes + 1)
                self.assertEqual(len(o.U_prev), o.nodes)
                self.assertEqual(len(o.q_sol), o.nodes + 1)
                self.assertEqual(len(o.v_sol), o.nodes)
                self.assertEqual(len(o.a_sol), o.nodes)
                self.assertTrue(np.all(np.isfinite(o.a_sol)))
                A = pin.computeCentroidalMap(o.model, o.robot.data, o.q_sol[0])
                np.testing.assert_allclose(A @ o.v_sol[0] / o.mass, o.x_nom[:6], atol=1e-6)
                # Final acceleration uses the last available joint velocity interval.
                dt = float(o.opti.value(o.dts[-2]))
                np.testing.assert_allclose(o.a_sol[-1][6:],
                                           (o.v_sol[-1][6:] - o.v_sol[-2][6:]) / dt, atol=1e-10)

    def test_arm_target_uses_configuration_after_momentum(self):
        o = self.make_problem()
        dq = np.zeros(o.nv)
        dq[5] = .25
        q = pin.integrate(o.model, o.robot.q0, dq)
        arm_target = np.array([.01, .02, 0.])
        o.set_tracking_targets(np.zeros(6), arm_target, np.zeros(3))
        x0 = np.r_[np.zeros(6), q]
        o.update_params(x0, 0.)
        o.solve(retract_all=False)
        velocity = np.asarray(o.dyn.get_frame_velocity(o.arm_ee_frame)(q, o.v_sol[0])).ravel()[:3]
        rotation = np.asarray(o.dyn.get_base_rotation()(q))
        expected = rotation @ arm_target
        expected[2] = arm_target[2]
        np.testing.assert_allclose(velocity, expected, atol=1e-6)
        # A change in momentum cannot change the initial configuration used for the target.
        initial_q = ca.Function("initial_q_test", [o.x_init], [o.get_initial_q()])
        np.testing.assert_allclose(np.asarray(initial_q(np.r_[np.arange(6), q])).ravel(), q)

    def test_one_qp_and_time_shifted_reference(self):
        o = self.make_problem()
        o.solve(retract_all=False)
        previous = o._prediction
        x_next = previous["states"][1]
        o.update_params(x_next, .015)
        np.testing.assert_allclose(o.opti.value(o.DX_opt[0], o.opti.initial()), 0., atol=1e-12)
        t = .030
        j = np.searchsorted(previous["grid"], t, side="right") - 1
        alpha = (t - previous["grid"][j]) / np.diff(previous["grid"])[j]
        xa, xb = previous["states"][j:j + 2]
        dx = o.opti.value(o.DX_opt[1], o.opti.initial())
        actual = np.asarray(o.dyn.state_integrate()(x_next, dx)).ravel()
        np.testing.assert_allclose(actual[:6], (1 - alpha) * xa[:6] + alpha * xb[:6])
        expected_q = pin.integrate(o.model, xa[6:], alpha * pin.difference(o.model, xa[6:], xb[6:]))
        np.testing.assert_allclose(actual[6:], expected_q, atol=1e-10)
        with patch("optimization.affine_qp.osqp.OSQP.solve", autospec=True, side_effect=osqp.OSQP.solve) as solve:
            with patch.object(o, "_armijo_line_search", side_effect=AssertionError("unexpected line search")):
                o.solve(retract_all=False)
            self.assertEqual(solve.call_count, 1)
        self.assertEqual(o.qp_solve_count, 2)
        self.assertLess(o.qp_constr_viol, SOLVER_ARGS["qp"]["max_qp_violation"])

    def test_contact_switch_and_friction_pyramid(self):
        o = self.make_problem(gait="trot")
        o.update_params(o.x_nom, .39)
        before = o.build_qp()
        o.solve(retract_all=False)
        x_next = o.dyn.state_integrate()(o.x_nom, o.DX_prev[1])
        o.update_params(x_next, .405)
        after = o.build_qp()
        self.assertEqual(o._pattern(before["A"]), o._pattern(after["A"]))
        o.solve(retract_all=False)
        schedule = np.asarray(o.opti.value(o.contact_schedule))
        for k, u in enumerate(o.U_prev):
            forces = u[o.f_idx:o.f_idx + 12].reshape(4, 3)
            np.testing.assert_allclose(forces[schedule[:, k] == 0], 0., atol=1e-5)
            self.assertTrue(np.all(np.abs(forces[:, 0]) + np.abs(forces[:, 1]) <= .9 * forces[:, 2] + 1e-5))

    def test_existing_centroidal_factory_and_solver_guard(self):
        o = self.make_problem()
        with self.assertRaisesRegex(ValueError, 'solver="qp"'):
            o.init_solver("osqp", SOLVER_ARGS["osqp"])
        old = make_ocp("centroidal_vel", {"include_base": True},
                       robot=o.robot, nodes=4, tau_nodes=2, warm_start=True)
        self.assertEqual(old.nx, o.nx)
        self.assertEqual(old.nu_opt, o.nu_opt)


if __name__ == "__main__":
    unittest.main()
