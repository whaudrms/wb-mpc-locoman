"""Centroidal velocity QP regressions, including reduced velocity inputs."""
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np
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

    def test_cached_sparse_layout_is_not_mutated_by_numeric_cleanup(self):
        o = self.make_problem()
        before = o.build_qp()
        pattern = o._pattern(before['A'])
        expected = before['A'].toarray()
        before['A'].eliminate_zeros()
        after = o.build_qp()
        self.assertEqual(o._pattern(after['A']), pattern)
        np.testing.assert_array_equal(after['A'].toarray(), expected)

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

    def test_one_qp_and_original_nlp_warm_start_policy(self):
        o = self.make_problem(gait="trot")
        o.solve(retract_all=False)
        # The reference must reuse the original deltas, even when the measured
        # state/time changes. Compare with the actual NLP routine, not a copy.
        old = make_ocp("centroidal_vel", {"include_base": True},
                       robot=o.robot, nodes=o.nodes, tau_nodes=o.tau_nodes, warm_start=True)
        old.set_time_params(.015, .04)
        old.set_swing_params(.07, [.1, -.2])
        old.set_tracking_targets(np.zeros(6), np.zeros(3), np.zeros(3))
        old.DX_prev = [x.copy() for x in o.DX_prev]
        old.U_prev = [u.copy() for u in o.U_prev]
        x_next = o._prediction["states"][1]
        old.update_params(x_next, .405)
        o.update_params(x_next, .405)
        for new, reference in zip(o.DX_opt + o.U_opt, old.DX_opt + old.U_opt):
            np.testing.assert_array_equal(o.opti.value(new, o.opti.initial()),
                                          old.opti.value(reference, old.opti.initial()))
        for state, previous in zip(o.DX_opt, o.DX_prev):
            np.testing.assert_array_equal(o.opti.value(state, o.opti.initial()), previous)
        with patch.object(o, "_solve_qpoases", wraps=o._solve_qpoases) as solve:
            with patch.object(o, "_armijo_line_search", side_effect=AssertionError("unexpected line search")):
                o.solve(retract_all=False)
            self.assertEqual(solve.call_count, 1)
        self.assertEqual(o.qp_solve_count, 2)
        self.assertLess(o.qp_constr_viol, 1e-3)

    def test_original_euler_transition_and_no_terminal_constraints(self):
        for include_base in (True, False):
            o = self.make_problem(include_base)
            rng = np.random.default_rng(62)
            for dx in o.DX_opt:
                o.opti.set_initial(dx, rng.normal(size=o.ndx_opt)*.03)
            for u in o.U_opt:
                o.opti.set_initial(u, rng.normal(size=o.nu_opt[0]))
            qp = o.build_qp()
            for k, rows in enumerate(o._qp_dynamics_rows):
                dx = np.asarray(o.opti.value(o.DX_opt[k], o.opti.initial())).ravel()
                next_dx = np.asarray(o.opti.value(o.DX_opt[k+1], o.opti.initial())).ravel()
                v = np.asarray(o.opti.value(o.get_v(k), o.opti.initial())).ravel()
                dt = float(o.opti.value(o.dts[k]))
                # Equality residual is -l, since l = lbg - g(reference).
                np.testing.assert_allclose(-qp['l'][rows[6:]], next_dx[6:]-dx[6:]-dt*v,
                                           atol=1e-10)
            terminal_start = o.nodes*(o.ndx_opt+o.nu_opt[0])
            non_transition = np.ones(qp['A'].shape[0], dtype=bool)
            non_transition[np.concatenate(o._qp_dynamics_rows)] = False
            self.assertEqual(qp['A'][non_transition, terminal_start:].nnz, 0)
            original = make_ocp("centroidal_vel", {"include_base": include_base},
                                robot=o.robot, nodes=o.nodes, tau_nodes=o.tau_nodes, warm_start=True)
            self.assertEqual(o.opti.nx, original.opti.nx)
            # Four pyramid faces replace one cone row at each foot/node.
            self.assertEqual(o.opti.ng-original.opti.ng, 3*o.n_feet*o.nodes)

    def test_disabled_warm_start_does_not_reuse_previous_solution(self):
        o = self.make_problem()
        before = np.asarray(o.opti.value(o._qp_z, o.opti.initial())).ravel()
        o.solve(retract_all=False)
        o.warm_start = False
        o.update_params(o._prediction['states'][1], .015)
        np.testing.assert_array_equal(o.opti.value(o._qp_z, o.opti.initial()), before)

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

    def test_stagewise_reconstruction_for_both_input_modes_and_gaits(self):
        rng = np.random.default_rng(31)
        for include_base in (True, False):
            for gait, free_per_stage in (("stand", 13), ("trot", 11)):
                with self.subTest(include_base=include_base, gait=gait):
                    o = self.make_problem(include_base, gait)
                    # Nonzero reference coordinates exercise the original delta-coordinate dynamics.
                    for k in range(1, o.nodes + 1):
                        o.opti.set_initial(o.DX_opt[k], rng.normal(size=o.ndx_opt) * .003)
                    qp = o.build_qp()
                    reduced, basis, offset = o.condense_qp(qp)
                    self.assertEqual(o.qp_condensing_method, "stagewise")
                    self.assertEqual(reduced["q"].size, free_per_stage * o.nodes)
                    self.assertIsNone(o.qp_condensing_fallback_reason)
                    eq = np.isfinite(qp["l"]) & (qp["l"] == qp["u"])
                    for y in (np.zeros(basis.shape[1]), rng.normal(size=basis.shape[1])):
                        np.testing.assert_allclose(qp["A"][eq] @ (offset + basis @ y),
                                                   qp["l"][eq], atol=1e-8)

    def test_rank_loss_falls_back_without_dropping_compatibility_equations(self):
        o = self.make_problem()
        qp = o.build_qp()
        row = o._qp_dynamics_rows[0][-1] + 1  # First momentum consistency row.
        qp["A"] = sparse.vstack((qp["A"], qp["A"][row]), format="csc")
        qp["l"] = np.r_[qp["l"], qp["l"][row]]
        qp["u"] = np.r_[qp["u"], qp["u"][row]]
        with patch.object(o, "build_qp", return_value=qp):
            o.solve(retract_all=False)
        self.assertEqual(o.qp_condensing_method, "global_qr")
        self.assertIn("row rank", o.qp_condensing_fallback_reason)
        self.assertEqual(o.qp_solve_count, 1)
        self.assertLess(o.qp_constr_viol, 1e-6)
        prediction = o._prediction
        qp["l"][-1] += 1.
        qp["u"][-1] += 1.
        with patch.object(o, "build_qp", return_value=qp):
            with self.assertRaisesRegex(ValueError, "Inconsistent affine equality"):
                o.solve(retract_all=False)
        self.assertIs(o._prediction, prediction)
        self.assertEqual(o.qp_solve_count, 1)

    def test_qpoases_iteration_failure_does_not_apply_prediction(self):
        o = self.make_problem()
        settings = dict(SOLVER_ARGS["qp"])
        settings["qpoases_opts"] = dict(settings["qpoases_opts"], nWSR=0)
        o.init_solver("qp", settings)
        with self.assertRaisesRegex(RuntimeError, "no solution applied"):
            o.solve(retract_all=False)
        self.assertIsNone(o._prediction)
        self.assertEqual(o.qp_solve_count, 1)
        self.assertEqual(len(o.q_sol), 0)

    def test_explicit_osqp_backend_and_invalid_backend(self):
        o = self.make_problem()
        settings = dict(SOLVER_ARGS["qp"], backend="osqp")
        o.init_solver("qp", settings)
        o.solve(retract_all=False)
        self.assertEqual(o.qp_backend, "osqp")
        self.assertEqual(o.qp_solve_count, 1)
        self.assertLess(o.qp_constr_viol, 1e-6)
        with self.assertRaisesRegex(ValueError, "Unknown QP backend"):
            o.init_solver("qp", dict(settings, backend="invalid"))

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
