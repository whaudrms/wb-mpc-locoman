"""qpOASES adapter checks against analytic QP solutions."""
import unittest

import numpy as np
from scipy import sparse

from args import SOLVER_ARGS
from optimization.affine_qp import AffineQPMixin


class BackendHarness(AffineQPMixin):
    qp_model_name = "backend_test"

    def __init__(self):
        self.qpoases_opts = dict(SOLVER_ARGS['qp']['qpoases_opts'], error_on_fail=False)
        self.qpoases_prob = None
        self._qpoases_shape = None


class QPOasesBackendTests(unittest.TestCase):
    def test_symmetric_hessian_and_repeated_numeric_updates(self):
        backend = BackendHarness()
        H = np.array([[4., 1.], [1., 2.]])
        qp = dict(P=sparse.csc_matrix(np.triu(H)), q=np.array([-1., -1.]),
                  A=sparse.csc_matrix([[1., 1.], [1., 0.], [0., 1.]]),
                  l=np.array([-np.inf, 0., 0.]), u=np.array([.4, np.inf, np.inf]))
        first_function = None
        for shift in (0., .2):
            qp['q'] = np.array([-1. - shift, -1.])
            x, success, info = backend._solve_qpoases(qp)
            self.assertTrue(success, info.status)
            kkt = np.block([[H, np.ones((2, 1))], [np.ones((1, 2)), np.zeros((1, 1))]])
            expected = np.linalg.solve(kkt, np.r_[-qp['q'], .4])[:2]
            np.testing.assert_allclose(x, expected, atol=1e-7)
            if first_function is None:
                first_function = backend.qpoases_prob
            else:
                self.assertIs(first_function, backend.qpoases_prob)
        # Contact/rank changes can change the condensed problem dimensions.
        smaller = dict(P=sparse.csc_matrix([[2.]]), q=np.array([-2.]),
                       A=sparse.csc_matrix([[1.]]), l=np.array([0.]), u=np.array([.5]))
        x, success, info = backend._solve_qpoases(smaller)
        self.assertTrue(success, info.status)
        self.assertIsNot(first_function, backend.qpoases_prob)
        np.testing.assert_allclose(x, [.5], atol=1e-8)

    def test_infeasible_constraints_are_not_reported_as_success(self):
        backend = BackendHarness()
        qp = dict(P=sparse.csc_matrix([[2.]]), q=np.zeros(1),
                  A=sparse.csc_matrix([[1.], [1.]]),
                  l=np.array([1., -np.inf]), u=np.array([np.inf, 0.]))
        _, success, _ = backend._solve_qpoases(qp)
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()
