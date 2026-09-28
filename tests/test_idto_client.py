"""Host integration test: requires the existing idto:latest Docker image."""
import shutil
import unittest

import numpy as np
from idto_client import IDTOClient


@unittest.skipUnless(shutil.which("docker"), "Docker CLI is required")
class ClientIntegrationTest(unittest.TestCase):
    def test_permuted_state_and_error_recovery(self):
        names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                 for joint in ("hip", "thigh", "calf")]
        names += [f"joint{i}" for i in range(1, 5)]
        with IDTOClient(nodes=8, threads=2, pin_joint_names=list(reversed(names))) as mpc:
            q, v, tau, info = mpc.solve(mpc.q0, mpc.v0, 0.)
            self.assertEqual(q.shape, (16,))
            self.assertEqual(v.shape, (16,))
            self.assertEqual(tau.shape, (16,))
            np.testing.assert_allclose(q, mpc.q0[7:], atol=1e-10)
            self.assertTrue(np.isfinite(np.r_[q, v, tau]).all())
            self.assertLess(info["base_residual_max"], 0.1)
            with self.assertRaisesRegex(RuntimeError, "Nonzero arm_force_des"):
                mpc.solve(mpc.q0, mpc.v0, 0.04, arm_force_des=[1, 0, 0])
            _, _, _, info = mpc.solve(mpc.q0, mpc.v0, 0.04)
            self.assertLess(info["base_residual_max"], 0.1)


if __name__ == "__main__":
    unittest.main()
