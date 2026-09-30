"""Compiled evaluation must preserve values, sparsity and cache identity."""
import tempfile
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

from optimization.qp_codegen import compile_functions, source_key


class QPCodegenTests(unittest.TestCase):
    def test_compiled_values_sparsity_and_cache_reuse(self):
        x=ca.SX.sym('x',3)
        f=ca.Function('codegen_test',[x],[ca.vertcat(x[0]*x[1],ca.sin(x[2])),
                                       ca.jacobian(ca.vertcat(x[0]*x[1],ca.sin(x[2])),x)])
        with tempfile.TemporaryDirectory() as directory:
            (compiled,), library = compile_functions([f],directory)
            for value in ([.2,-.5,.8],[0.,3.,-2.]):
                for a,b in zip(f(value),compiled(value)):
                    self.assertEqual(a.sparsity(),b.sparsity())
                    np.testing.assert_allclose(a.nonzeros(),b.nonzeros(),atol=1e-14)
            with patch('optimization.qp_codegen.subprocess.run',side_effect=AssertionError('cache miss')):
                _, same_library = compile_functions([f],directory)
            self.assertEqual(library,same_library)

    def test_cache_ignores_comments_but_not_numerical_code(self):
        self.assertEqual(source_key(b'/* opti0 */ double a=1;'),source_key(b'/* opti9 */ double a=1;'))
        self.assertNotEqual(source_key(b'double a=1;'),source_key(b'double a=2;'))


if __name__=='__main__':unittest.main()
