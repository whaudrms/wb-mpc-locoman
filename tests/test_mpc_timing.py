"""Check identical MPC timing boundaries without relying on noisy wall time."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import main


class Clock:
    now = 0.

    def advance(self, seconds):
        self.now += seconds


class TimedOCP:
    def __init__(self, clock):
        self.clock = clock
        self.x_nom = np.zeros(1)
        self.DX_prev = [np.zeros(1), np.zeros(1)]
        self.dts = [.015]
        self.opti = SimpleNamespace(p=np.zeros(1), value=lambda x: x)
        self.dyn = SimpleNamespace(state_integrate=lambda: self.integrate)
        self.solver_function = self.nlp_solve

    def update_params(self, state, t):
        self.clock.advance(1.)

    def init_solver(self, *args):
        self.clock.advance(100.)  # Initialization must not enter step timing.

    def get_solver_params(self):
        self.clock.advance(2.)
        return []

    def nlp_solve(self):
        self.clock.advance(3.)
        return np.zeros(1)

    def g_data(self, *args):
        self.clock.advance(4.)
        return np.zeros(1), np.zeros(1), np.zeros(1)

    def constr_viol_norm_inf(self, *args):
        return 0.

    def retract_stacked_sol(self, *args, **kwargs):
        self.clock.advance(5.)

    def integrate(self, state, delta):
        self.clock.advance(6.)
        return state

    def solve(self, retract_all, verbose):
        assert not retract_all and not verbose
        self.get_solver_params()
        self.nlp_solve()
        self.solve_time = 5.
        self.g_data()
        self.retract_stacked_sol()
        self.constr_viol = 0.

    def print_qp_stats(self):
        self.clock.advance(30.)  # QP diagnostics must also be outside timing.
        print("QP diagnostics")


class MPCTimingTests(unittest.TestCase):
    def test_same_full_step_boundaries_for_qp_fatrop_ipopt(self):
        for solver in ('qp', 'fatrop', 'ipopt'):
            for log_mode in ('original', 'detailed'):
                with self.subTest(solver=solver, log_mode=log_mode):
                    clock = Clock()
                    ocp = TimedOCP(clock)
                    output = io.StringIO()
                    with patch.multiple(main, mpc_loops=2, compile_solver=False,
                                        load_compiled_solver=None, log_mode=log_mode):
                        with patch.object(main, 'get_solver_configuration', return_value=(solver, {})):
                            with patch.object(main.time, 'perf_counter', side_effect=lambda: clock.now):
                                with contextlib.redirect_stdout(output):
                                    main.mpc_loop(ocp)
                    np.testing.assert_array_equal(ocp.mpc_step_times, [21., 21.])
                    np.testing.assert_array_equal(ocp.optimization_times, [5., 5.])
                    np.testing.assert_array_equal(ocp.mpc_constraint_violations, [0., 0.])
                    solve_seconds = 5. if solver == 'qp' else 3.
                    np.testing.assert_array_equal(ocp.solve_times, [solve_seconds] * 2)
                    log = output.getvalue()
                    if log_mode == 'original':
                        self.assertEqual(log.count('Solve time (ms): '), 2)
                        self.assertIn(f'Avg solve time (ms):  {solve_seconds * 1000}', log)
                        self.assertIn('Std solve time (ms): ', log)
                        self.assertNotIn('MPC step:', log)
                        self.assertNotIn('QP diagnostics', log)
                    else:
                        self.assertEqual(log.count('MPC step: '), 2)
                        self.assertIn('Avg full MPC step (ms):  21000.0', log)
                        self.assertIn('Avg optimization time (ms):  5000.0', log)
                        self.assertNotIn('Avg solve time (ms): ', log)
                        self.assertEqual(log.count('QP diagnostics'), 2 if solver == 'qp' else 0)


if __name__ == '__main__':
    unittest.main()
