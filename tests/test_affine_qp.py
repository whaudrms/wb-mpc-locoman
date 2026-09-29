"""Algebraic equivalence checks for the optional equality condensation."""
import unittest

import numpy as np
from scipy import sparse

from optimization.affine_qp import condense_equalities


class EqualityCondensationTests(unittest.TestCase):
    def problem(self):
        return dict(P=sparse.csc_matrix([[2., .3, 0.], [0., 4., .1], [0., 0., 6.]]),
                    q=np.array([.1, 2., 3.]),
                    A=sparse.csc_matrix([[1., 1., 0.], [2., 2., 0.], [0., 1., -1.]]),
                    l=np.array([1., 2., -.4]), u=np.array([1., 2., .8]))

    def test_equalities_inequalities_and_objective_are_preserved(self):
        qp = self.problem()
        reduced, basis, offset = condense_equalities(qp)
        self.assertEqual(basis.shape, (3, 2))
        H = qp["P"] + qp["P"].T - sparse.diags(qp["P"].diagonal())
        Hr = reduced["P"] + reduced["P"].T - sparse.diags(reduced["P"].diagonal())
        constant = .5 * offset @ H @ offset + qp["q"] @ offset
        self.assertGreater(np.linalg.eigvalsh(Hr.toarray()).min(), 0.)
        for y in (np.zeros(2), np.array([.2, -.7]), np.array([-3., 2.])):
            z = offset + basis @ y
            np.testing.assert_allclose((qp["A"] @ z)[:2], qp["l"][:2], atol=1e-12)
            np.testing.assert_allclose(reduced["A"] @ y - reduced["l"],
                                       (qp["A"] @ z)[2:] - qp["l"][2:], atol=1e-12)
            np.testing.assert_allclose(reduced["u"] - reduced["A"] @ y,
                                       qp["u"][2:] - (qp["A"] @ z)[2:], atol=1e-12)
            original_cost = .5 * z @ H @ z + qp["q"] @ z
            reduced_cost = .5 * y @ Hr @ y + reduced["q"] @ y
            self.assertAlmostEqual(original_cost - constant, reduced_cost, places=10)

    def test_inconsistent_redundant_equality_is_rejected(self):
        qp = self.problem()
        qp["l"][1] = qp["u"][1] = 3.
        with self.assertRaisesRegex(ValueError, "Inconsistent affine equality"):
            condense_equalities(qp)

    def test_fully_determined_primal_is_reconstructed(self):
        qp = dict(P=sparse.eye(2, format="csc"), q=np.zeros(2),
                  A=sparse.eye(2, format="csc"), l=np.array([2., -1.]), u=np.array([2., -1.]))
        reduced, basis, offset = condense_equalities(qp)
        self.assertEqual(reduced["P"].shape, (1, 1))
        np.testing.assert_allclose(offset + basis @ np.array([5.]), qp["l"])


if __name__ == "__main__":
    unittest.main()
