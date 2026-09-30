"""Independent analytic checks for benchmark conversion and accuracy metrics."""
import unittest

import numpy as np
from scipy import sparse

from benchmarks.qp_solver_adapters import ADAPTERS, canonical_qp, quality


def hpipm_available():
    from optimization.hpipm_backend import load_hpipm
    try:
        load_hpipm()
    except RuntimeError:
        return False
    return True


class BenchmarkAdapterTests(unittest.TestCase):
    def problem(self, equality=False):
        return dict(P=sparse.csc_matrix([[4., 1.], [0., 2.]]), q=np.array([-1., -1.]),
                    A=sparse.csc_matrix([[1., 1.], [1., 0.], [0., 1.], [0., 0.]]),
                    l=np.array([.4 if equality else -np.inf, 0., 0., -np.inf]),
                    u=np.array([.4, np.inf, np.inf, -1e-13]))

    def check_solver(self, name):
        for equality in (False, True):
            with self.subTest(equality=equality):
                qp = canonical_qp(self.problem(equality))
                solver = ADAPTERS[name]()
                result = solver.solve(qp)
                self.assertTrue(result['success'], result['status'])
                np.testing.assert_allclose(result['x'], [.1, .3], atol=1e-6)
                q = quality(qp, result)
                self.assertLess(q['primal'], 1e-7)
                self.assertLess(q['scaled_stationarity'], 1e-7)
                self.assertLess(q['scaled_complementarity'], 1e-7)

    def test_osqp_and_qpoases_conversion(self):
        for name in ('osqp', 'qpoases'):
            with self.subTest(solver=name):
                self.check_solver(name)

    @unittest.skipUnless(hpipm_available(), 'Optional HPIPM build not on library/Python paths')
    def test_hpipm_bounds_masks_and_equality_dual_sign(self):
        self.check_solver('hpipm')

    @unittest.skipUnless(hpipm_available(), 'Optional HPIPM build not on library/Python paths')
    def test_ocp_transition_scaling_cross_cost_and_dual_recovery(self):
        from benchmarks.hpipm_ocp_adapter import HPIPMOCPAdapter
        # x0=1, 2*x1-2*x0-2*u0=0, u0>=-.25. The unconstrained
        # minimizer violates the input bound; the exact optimum is known.
        qp = dict(P=sparse.csc_matrix([[2., .2, 0.], [0., 4., 0.], [0., 0., 2.]]),
                  q=np.zeros(3),
                  A=sparse.csc_matrix([[1., 0., 0.], [-2., -2., 2.],
                                       [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]]),
                  l=np.array([1., 0., -.25, -np.inf, -np.inf]),
                  u=np.array([1., 0., np.inf, 2., 0.]))
        solver = HPIPMOCPAdapter(1, 1, [np.array([1])])
        for _ in range(2):  # includes workspace reuse
            result = solver.solve(qp)
            self.assertTrue(result['success'], result)
            np.testing.assert_allclose(result['x'], [1., -.25, .75], atol=1e-7)
            H = qp['P']+qp['P'].T-sparse.diags(qp['P'].diagonal())
            metrics = quality(dict(qp, H=H), result)
            self.assertLess(metrics['primal'], 1e-7)
            self.assertLess(metrics['scaled_stationarity'], 1e-7)
            self.assertLess(metrics['scaled_complementarity'], 1e-7)

    def test_zero_row_cleanup_rejects_meaningful_contradiction(self):
        qp = self.problem()
        self.assertEqual(canonical_qp(qp)['A'].shape, (3, 2))
        qp['u'][-1] = -.01
        with self.assertRaisesRegex(ValueError, 'Inconsistent zero'):
            canonical_qp(qp)

    def test_infeasible_osqp_object_array_is_not_a_numeric_solution(self):
        qp = canonical_qp(self.problem())
        r = dict(x=np.array([None, None], dtype=object), dual=np.array([None]*3, dtype=object))
        self.assertFalse(quality(qp, r)['finite'])


if __name__ == '__main__':
    unittest.main()
