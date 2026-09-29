"""Run from the repository root: python -m unittest discover -s tests -v."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pinocchio as pin
from scipy import sparse

from args import DYN_ARGS, SOLVER_ARGS
from optimization import make_ocp
from utils.robot import B2_Z1


class WholeBodyQPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = B2_Z1(reference_pose="standing_with_arm_up", arm_joints=4)
        cls.robot.set_gait_sequence("stand", .8)

    def make_problem(self, warm_start=True):
        o = make_ocp("whole_body_rnea_qp", DYN_ARGS["whole_body_rnea_qp"],
                     robot=self.robot, nodes=4, tau_nodes=1, warm_start=warm_start)
        o.set_time_params(.015, .04)
        o.set_swing_params(.07, [.1, -.2])
        o.set_tracking_targets(np.zeros(6), np.zeros(3), np.zeros(3))
        o.update_params(o.x_nom, 0.)
        o.init_solver("qp", SOLVER_ARGS["qp"])
        return o

    def test_convexity_and_constraint_derivative(self):
        o = self.make_problem()
        qp = o.build_qp()
        self.assertEqual(o.tau_nodes, o.nodes)
        self.assertTrue(all(n == o.nv + o.nf + o.nj for n in o.nu_opt))
        # The exact quadratic tracking Hessian is diagonal and positive after regularization.
        self.assertEqual((qp["P"] - sparse.diags(qp["P"].diagonal())).nnz, 0)
        self.assertGreater(qp["P"].diagonal().min(), 0.)
        rng = np.random.default_rng(42)
        direction = rng.normal(size=o.opti.nx)
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        g_plus = np.asarray(o.g_data(o.qp_reference + eps * direction, o.qp_params)[0]).ravel()
        g_minus = np.asarray(o.g_data(o.qp_reference - eps * direction, o.qp_params)[0]).ravel()
        np.testing.assert_allclose((g_plus - g_minus) / (2 * eps), qp["A"] @ direction,
                                   atol=2e-6, rtol=2e-5)

    def test_one_solve_and_physical_bounds(self):
        o = self.make_problem()
        o.solve(retract_all=False)
        x_next = o.dyn.state_integrate()(o.x_nom, o.DX_prev[1])
        o.update_params(x_next, .015)
        np.testing.assert_allclose(o.opti.value(o.DX_opt[0], o.opti.initial()), 0., atol=1e-12)
        with patch.object(o.osqp_prob, "solve", wraps=o.osqp_prob.solve) as solve:
            with patch.object(o, "_armijo_line_search", side_effect=AssertionError("unexpected line search")):
                o.solve(retract_all=False)
            self.assertEqual(solve.call_count, 1)
        self.assertEqual(o.qp_solve_count, 2)
        self.assertLess(o.qp_constr_viol, SOLVER_ARGS["qp"]["max_qp_violation"])
        for u in o.U_prev:
            forces = u[o.f_idx:o.tau_idx][:12].reshape(4, 3)
            self.assertTrue(np.all(forces[:, 2] >= -1e-5))
            self.assertTrue(np.all(np.abs(forces[:, 0]) + np.abs(forces[:, 1]) <= .9 * forces[:, 2] + 1e-5))
            self.assertTrue(np.all(np.abs(u[o.tau_idx:]) <= self.robot.joint_torque_max + 1e-5))
        x_terminal = np.asarray(o.dyn.state_integrate()(x_next, o.DX_prev[-1])).ravel()
        self.assertTrue(np.all(x_terminal[7:o.nq] <= self.robot.joint_pos_max + 1e-5))
        self.assertTrue(np.all(x_terminal[7:o.nq] >= self.robot.joint_pos_min - 1e-5))

    def test_warm_start_time_shift_and_reanchor(self):
        o = self.make_problem()
        o.solve(retract_all=False)
        previous = o._prediction
        x_next = previous["states"][1]
        o.update_params(x_next, .015)
        # New node 1 lies between previous nodes 1 and 2 on a nonuniform grid.
        t = .030
        j = np.searchsorted(previous["grid"], t, side="right") - 1
        alpha = (t - previous["grid"][j]) / np.diff(previous["grid"])[j]
        xa, xb = previous["states"][j:j + 2]
        expected_q = pin.integrate(o.model, xa[:o.nq], alpha * pin.difference(o.model, xa[:o.nq], xb[:o.nq]))
        dx = o.opti.value(o.DX_opt[1], o.opti.initial())
        actual = np.asarray(o.dyn.state_integrate()(x_next, dx)).ravel()
        np.testing.assert_allclose(actual[:o.nq], expected_q, atol=1e-10)
        np.testing.assert_allclose(actual[o.nq:], (1 - alpha) * xa[o.nq:] + alpha * xb[o.nq:], atol=1e-10)

    def test_failed_qp_does_not_apply_a_solution(self):
        o = self.make_problem()
        o.solve(retract_all=False)
        before = o.last_solution.copy()
        prediction = o._prediction
        count = len(o.q_sol)
        failed = SimpleNamespace(info=SimpleNamespace(status="primal infeasible", status_val=3), x=None)
        with patch.object(o.osqp_prob, "solve", return_value=failed) as solve:
            with self.assertRaisesRegex(RuntimeError, "no solution applied"):
                o.solve(retract_all=False)
            self.assertEqual(solve.call_count, 1)
        np.testing.assert_array_equal(o.last_solution, before)
        self.assertIs(o._prediction, prediction)
        self.assertEqual(len(o.q_sol), count)

    def test_invalid_weights_and_solver_are_rejected(self):
        o = self.make_problem()
        with self.assertRaisesRegex(ValueError, 'solver="qp"'):
            o.init_solver("osqp", SOLVER_ARGS["osqp"])
        w = np.asarray(o.opti.value(o.Q_diag)).ravel()
        w[0] = -1.
        o.opti.set_value(o.Q_diag, w)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            o.build_qp()

    def test_contact_switch_preserves_structure_and_zero_swing_forces(self):
        robot = B2_Z1(reference_pose="standing_with_arm_up", arm_joints=4)
        robot.set_gait_sequence("trot", .8)
        o = make_ocp("whole_body_rnea_qp", DYN_ARGS["whole_body_rnea_qp"],
                     robot=robot, nodes=4, tau_nodes=1, warm_start=True)
        o.set_time_params(.015, .04)
        o.set_swing_params(.07, [.1, -.2])
        o.set_tracking_targets(np.zeros(6), np.zeros(3), np.zeros(3))
        o.update_params(o.x_nom, .39)
        o.init_solver("qp", SOLVER_ARGS["qp"])
        before = o.build_qp()
        contact_before = np.asarray(o.opti.value(o.contact_schedule))
        o.solve(retract_all=False)
        x_next = o.dyn.state_integrate()(o.x_nom, o.DX_prev[1])
        o.update_params(x_next, .405)
        after = o.build_qp()
        self.assertEqual(o._pattern(before["A"]), o._pattern(after["A"]))
        schedule = np.asarray(o.opti.value(o.contact_schedule))
        self.assertTrue(np.any(contact_before[:, 0] != schedule[:, 0]))
        for k in range(o.nodes):
            u = np.asarray(o.opti.value(o.U_opt[k], o.opti.initial())).ravel()
            forces = u[o.f_idx:o.f_idx + 12].reshape(4, 3)
            np.testing.assert_array_equal(forces[schedule[:, k] == 0], 0.)
        o.solve(retract_all=False)
        for k, u in enumerate(o.U_prev):
            forces = u[o.f_idx:o.f_idx + 12].reshape(4, 3)
            np.testing.assert_allclose(forces[schedule[:, k] == 0], 0., atol=1e-4)

    def test_existing_rnea_factory_is_preserved(self):
        o = make_ocp("whole_body_rnea", DYN_ARGS["whole_body_rnea"],
                     robot=self.robot, nodes=4, tau_nodes=1, warm_start=True)
        self.assertEqual(o.tau_nodes, 1)
        self.assertNotEqual(o.nu_opt[0], o.nu_opt[-1])


if __name__ == "__main__":
    unittest.main()
