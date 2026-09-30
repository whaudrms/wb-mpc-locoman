"""Exercise main.py's six QP configurations through real controller solves."""
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

import main
from args import SOLVER_ARGS
from optimization import make_ocp
from optimization.hpipm_backend import load_hpipm
from utils.robot import B2_Z1


def hpipm_available():
    try:
        load_hpipm()
        return True
    except RuntimeError:
        return False


class QPConfigurationTests(unittest.TestCase):
    def problem(self, backend, condensed, include_base=True):
        robot = B2_Z1(reference_pose='standing_with_arm_up', arm_joints=4)
        robot.set_gait_sequence('trot', .8)
        o = make_ocp('centroidal_vel_qp', {'include_base': include_base},
                     robot=robot, nodes=4, tau_nodes=2, warm_start=True)
        o.set_time_params(.015, .04)
        o.set_swing_params(.07, [.1, -.2])
        o.set_tracking_targets(main.base_vel_des, main.arm_vel_des, main.arm_force_des)
        o.update_params(o.x_nom, .39)
        with patch.multiple(main, dynamics='centroidal_vel_qp', solver=backend, qp_condesed=condensed):
            kind, options = main.get_solver_configuration()
        o.init_solver(kind, dict(options, compile_data=False))  # Native evaluation has separate tests.
        return o

    def check_combinations(self, names):
        for name in names:
            for condensed in (False, True):
                with self.subTest(solver=name, condensed=condensed):
                    o = self.problem(name, condensed)
                    for k in range(2):
                        with contextlib.redirect_stdout(io.StringIO()):
                            o.solve(retract_all=False)
                        self.assertEqual(o.qp_solve_count, k+1)
                        self.assertLess(o.qp_constr_viol, 1e-3)
                        if condensed:
                            self.assertLess(o.last_solver_qp['q'].size, o.last_qp['q'].size)
                            self.assertEqual(o.qp_condensing_method, 'stagewise')
                        else:
                            self.assertEqual(o.last_solver_qp['q'].size, o.last_qp['q'].size)
                            self.assertEqual(o.qp_condensing_method, 'none')
                        x_next = o._prediction['states'][1]
                        o.update_params(x_next, .405+k*.015)
                    if name=='HPIPM':
                        self.assertEqual(o.qp_hpipm_form, 'dense' if condensed else 'ocp')

    def test_osqp_and_qpoases_condensing_on_off_with_contact_switch(self):
        self.check_combinations(('qpOASE', 'OSQP'))

    @unittest.skipUnless(hpipm_available(), 'Optional HPIPM build unavailable')
    def test_hpipm_condensing_on_off_with_contact_switch(self):
        self.check_combinations(('HPIPM',))

    @unittest.skipUnless(hpipm_available(), 'Optional HPIPM build unavailable')
    def test_hpipm_reduced_base_velocity_and_failed_solve(self):
        for condensed in (False, True):
            with self.subTest(condensed=condensed):
                o = self.problem('HPIPM', condensed, include_base=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    o.solve(retract_all=False)
                self.assertLess(o.qp_constr_viol, 1e-3)
                previous = o._prediction
                count = len(o.q_sol)
                # A real native iteration-limit failure must not update outputs.
                options = dict(SOLVER_ARGS['qp'], backend='hpipm', condensed=condensed,
                               hpipm_opts=dict(SOLVER_ARGS['qp']['hpipm_opts'], iter_max=0))
                o.init_solver('qp', options)
                with self.assertRaisesRegex(RuntimeError, 'no solution applied'):
                    o.solve(retract_all=False)
                self.assertIs(o._prediction, previous)
                self.assertEqual(len(o.q_sol), count)
                self.assertEqual(o.qp_solve_count, 1)

    def test_legacy_configuration_and_input_validation(self):
        with patch.multiple(main, dynamics='whole_body_rnea_qp', solver='qp', qp_condesed=False):
            kind, options = main.get_solver_configuration()
            self.assertEqual(kind, 'qp')
            self.assertNotIn('backend', options)
            self.assertFalse(options['condensed'])
        with patch.multiple(main, dynamics='centroidal_vel', solver='osqp'):
            kind, options = main.get_solver_configuration()
            self.assertEqual(kind, 'osqp')
            self.assertIs(options, SOLVER_ARGS['osqp'])
        with patch.multiple(main, dynamics='centroidal_vel', solver='HPIPM'):
            with self.assertRaisesRegex(ValueError, 'Nonlinear dynamics'):
                main.get_solver_configuration()
        with self.assertRaisesRegex(ValueError, 'True or False'):
            self.problem('OSQP', 'false')
        with patch.multiple(main, dynamics='centroidal_vel_qp', solver='typo'):
            with self.assertRaisesRegex(ValueError, 'requires solver'):
                main.get_solver_configuration()

    def test_missing_explicit_hpipm_install_has_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, 'build_hpipm.sh'):
            load_hpipm('/nonexistent/wb-mpc-test-hpipm')


if __name__ == '__main__':
    unittest.main()
