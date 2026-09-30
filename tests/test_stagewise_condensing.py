"""Cost/feasible-set preservation and exceptional stage structures."""
import unittest

import numpy as np
from scipy import sparse

from optimization.affine_qp import condense_equalities
from optimization.qp_condensing import condense_stagewise, StageCondensingError


class StagewiseCondensingTests(unittest.TestCase):
    def problem(self):
        # Nonidentity transition Jacobians, affine defects, nonzero fixed x0,
        # two stages, and cross-stage objective terms.
        rng = np.random.default_rng(19)
        nx, nu, N = 2, 3, 2
        n = (nx + nu) * N + nx
        A = np.zeros((10, n))
        A[:2, :2] = np.array([[1., .2], [0., 2.]])
        for k in range(N):
            c = k * 5
            r = 2 + 3 * k
            A[r:r + 2, c:c + 7] = rng.normal(size=(2, 7))
            A[r:r + 2, c + 5:c + 7] = np.array([[2., .1], [.3, 1.]])
            A[r + 2, c:c + 5] = rng.normal(size=5)
        A[8:] = rng.normal(size=(2, n))
        rhs = rng.normal(size=8)
        lower = np.r_[rhs, -2., -np.inf]
        upper = np.r_[rhs, 3., 4.]
        W = rng.normal(size=(n, n))
        H = W.T @ W + np.eye(n)
        return dict(P=sparse.csc_matrix(np.triu(H)), q=rng.normal(size=n),
                    A=sparse.csc_matrix(A), l=lower, u=upper)

    def test_nonzero_defects_objective_bounds_and_all_freedoms_are_preserved(self):
        qp = self.problem()
        reduced, T, d = condense_stagewise(qp, 2, 3, [[2, 3], [5, 6]])
        _, general_T, _ = condense_equalities(qp)
        self.assertEqual(T.shape, general_T.shape)
        self.assertEqual(np.linalg.matrix_rank(T), T.shape[1])
        self.assertEqual(T.shape[1], 4)
        H = qp['P'] + qp['P'].T - sparse.diags(qp['P'].diagonal())
        Hr = reduced['P'] + reduced['P'].T - sparse.diags(reduced['P'].diagonal())
        np.testing.assert_allclose(Hr.diagonal(), 1., atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(Hr.toarray()).min(), 0.)
        constant = .5 * d @ H @ d + qp['q'] @ d
        for y in np.random.default_rng(8).normal(size=(5, 4)):
            z = d + T @ y
            np.testing.assert_allclose((qp['A'] @ z)[:8], qp['l'][:8], atol=1e-12)
            np.testing.assert_allclose(reduced['A'] @ y - reduced['l'],
                                       (qp['A'] @ z)[8:] - qp['l'][8:], atol=1e-12)
            np.testing.assert_allclose(reduced['u'] - reduced['A'] @ y,
                                       qp['u'][8:] - (qp['A'] @ z)[8:], atol=1e-12)
            self.assertAlmostEqual(.5*z@H@z + qp['q']@z - constant,
                                   .5*y@Hr@y + reduced['q']@y, places=10)

    def test_rank_loss_preserves_need_for_state_compatibility(self):
        qp = self.problem()
        A = qp['A'].toarray()
        A[4, 2:5] = 0.  # This now constrains x0; it cannot just be dropped.
        qp['A'] = sparse.csc_matrix(A)
        with self.assertRaisesRegex(StageCondensingError, 'row rank'):
            condense_stagewise(qp, 2, 3, [[2, 3], [5, 6]])

    def test_zero_rows_are_removed_but_contradictions_are_rejected(self):
        qp = self.problem()
        qp['A'] = sparse.vstack((qp['A'], sparse.csc_matrix((1, 12))), format='csc')
        qp['l'] = np.r_[qp['l'], 0.]
        qp['u'] = np.r_[qp['u'], 0.]
        condense_stagewise(qp, 2, 3, [[2, 3], [5, 6]])
        qp['l'][-1] = qp['u'][-1] = 1.
        with self.assertRaisesRegex(ValueError, 'Inconsistent affine equality'):
            condense_stagewise(qp, 2, 3, [[2, 3], [5, 6]])

    def test_singular_next_state_block_requires_general_reduction(self):
        qp = self.problem()
        A = qp['A'].toarray()
        A[2:4, 5:7] = 0.
        qp['A'] = sparse.csc_matrix(A)
        with self.assertRaisesRegex(StageCondensingError, 'Singular'):
            condense_stagewise(qp, 2, 3, [[2, 3], [5, 6]])

    def test_fully_determined_and_unconstrained_inputs(self):
        qp = dict(P=sparse.eye(3, format='csc'), q=np.zeros(3),
                  A=sparse.csc_matrix([[1., 0., 0.], [-1., -1., 1.], [0., 1., 0.]]),
                  l=np.array([2., 1., 3.]), u=np.array([2., 1., 3.]))
        reduced, T, d = condense_stagewise(qp, 1, 1, [[1]])
        self.assertEqual(reduced['P'].shape, (1, 1))
        np.testing.assert_allclose(d + T @ [7.], [2., 3., 6.])
        qp['l'][2], qp['u'][2] = -10., 10.
        reduced, T, d = condense_stagewise(qp, 1, 1, [[1]])
        np.testing.assert_allclose((qp['A'] @ (d + T @ [4.]))[:2], qp['l'][:2])


if __name__ == '__main__':
    unittest.main()
