"""Run with bash scripts/run_idto.sh --test inside the matched Drake runtime."""
import unittest
import numpy as np
from pydrake.math import RollPitchYaw
from idto_mpc import B2Z1MPC, MPCConfig, StateAdapter, build_model
from idto_mpc.model import joint_names


class ModelAndStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        builder, cls.plant, cls.sg, _ = build_model(0.04)
        cls.diagram = builder.Build()
        cls.context = cls.diagram.CreateDefaultContext()
        cls.pc = cls.plant.GetMyMutableContextFromRoot(cls.context)
        cls.adapter = StateAdapter(cls.plant, joint_names())

    def test_reduced_model_and_actuation(self):
        p = self.plant
        self.assertEqual((p.num_positions(), p.num_velocities(), p.num_actuators()), (23, 22, 16))
        self.assertEqual(p.GetJointByName("joint5").num_positions(), 0)
        self.assertEqual(p.GetJointByName("joint6").num_positions(), 0)
        self.assertEqual(p.GetJointByName("jointGripper").num_positions(), 0)
        B = p.MakeActuationMatrix()
        np.testing.assert_allclose(B.T @ B, np.eye(16))
        self.assertEqual(np.count_nonzero(np.sum(B, axis=1) == 0), 6)
        self.assertEqual(p.num_collision_geometries(), 5)

    def test_known_rotated_base_velocity(self):
        a = self.adapter
        quat = RollPitchYaw(0., 0., np.pi / 2).ToQuaternion().wxyz()
        qp = np.r_[[1., 2., 3.], quat[[1, 2, 3, 0]], np.arange(16) / 10]
        vp = np.r_[[1., 0., 0., 0., 2., 0.], np.arange(16) / 5]
        q, v = a.to_drake(qp, vp)
        np.testing.assert_allclose(q[a.qb:a.qb + 4], quat)
        np.testing.assert_allclose(v[a.vb:a.vb + 6], [-2, 0, 0, 0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(q[a.qj], qp[7:])
        qr, vr = a.to_pin(q, v)
        np.testing.assert_allclose(qr, qp, atol=1e-12)
        np.testing.assert_allclose(vr, vp, atol=1e-12)

    def test_joint_name_permutation_and_torque(self):
        a = StateAdapter(self.plant, tuple(reversed(joint_names())))
        qp = np.r_[[1., 2., 3., 0., 0., 0., 1.], np.arange(16)]
        vp = np.r_[np.zeros(6), np.arange(16)]
        q, v = a.to_drake(qp, vp)
        for i, name in enumerate(a.names):
            j = self.plant.GetJointByName(name)
            self.assertEqual(q[j.position_start()], i)
            self.assertEqual(v[j.velocity_start()], i)
        applied = a.B @ a.actuator_torques(np.arange(16))
        np.testing.assert_allclose(applied[a.vj], np.arange(16))
        np.testing.assert_allclose(applied[a.vb:a.vb + 6], np.zeros(6))

    def test_invalid_state_rejected(self):
        a = self.adapter
        with self.assertRaises(ValueError):
            a.to_drake(np.zeros(23), np.zeros(22))
        with self.assertRaises(ValueError):
            a.to_drake(np.ones(24), np.zeros(22))
        with self.assertRaises(ValueError):
            StateAdapter(self.plant, ["joint1"] * 16)
        with self.assertRaises(ValueError):
            MPCConfig(dt=0.)

    def test_zero_and_six_arm_dof_models(self):
        for n in (0, 6):
            builder, plant, _, _ = build_model(0.04, n)
            diagram = builder.Build()
            context = diagram.CreateDefaultContext()
            a = StateAdapter(plant, joint_names(n))
            q, v = a.standing_state(plant.GetMyMutableContextFromRoot(context))
            self.assertEqual(q.shape, (19 + n,))
            self.assertEqual(v.shape, (18 + n,))
            self.assertEqual(plant.num_actuators(), 12 + n)


class OptimizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mpc = B2Z1MPC(MPCConfig(nodes=8, initial_iterations=10, threads=2))

    def test_unsupported_commands_rejected(self):
        m = self.mpc
        with self.assertRaises(NotImplementedError):
            m.solve(m.q0_pin, m.v0_pin, 1., arm_force_des=[1, 0, 0])
        with self.assertRaises(ValueError):
            m.solve(m.q0_pin, m.v0_pin, 1., base_vel_des=[0, 0, 1, 0, 0, 0])

    def test_solve_feedback_and_warm_start(self):
        m = self.mpc
        diag = m.solve(m.q0_pin, m.v0_pin, 0.)
        self.assertLess(diag["base_residual_max"], 0.1)
        self.assertEqual(diag["joint_position_violation"], 0.)
        self.assertEqual(diag["joint_velocity_violation"], 0.)
        self.assertEqual(diag["joint_torque_violation"], 0.)
        qp, vp = m.adapter.to_pin(m.solution[0][1], m.solution[1][1])
        qp[0] += 0.002  # A measured disturbance, not just the predicted state.
        qp[3:7] *= -1  # Equivalent orientation with a different quaternion sign.
        diag = m.solve(qp, vp, 0.04)
        qd, vd = m.adapter.to_drake(qp, vp)
        if qd[m.adapter.qb:m.adapter.qb + 4] @ m.solution[0][0, m.adapter.qb:m.adapter.qb + 4] < 0:
            qd[m.adapter.qb:m.adapter.qb + 4] *= -1
        np.testing.assert_allclose(m.solution[0][0], qd, atol=1e-10)
        np.testing.assert_allclose(m.solution[1][0], vd, atol=1e-10)
        self.assertLess(diag["base_residual_max"], 1.)
        for command in m.command(0.045):
            self.assertEqual(command.shape, (16,))
            self.assertTrue(np.isfinite(command).all())
        with self.assertRaises(RuntimeError):
            m.command(10.)
        with self.assertRaises(ValueError):
            m.solve(qp, vp, -1.)

    def test_arm_target_tracks_relative_velocity(self):
        m = self.mpc
        qs, vs = m._nominal(m.q0, np.zeros(6), np.array([0.01, 0., 0.]))
        frame = m.plant.GetJointByName("gripperCenter").frame_on_child()
        points = []
        for q in qs[:2]:
            m.plant.SetPositions(m.plant_context, q)
            points.append(m.plant.CalcPointsPositions(m.plant_context, frame, np.zeros(3),
                                                      m.plant.world_frame()).ravel())
        np.testing.assert_allclose((points[1] - points[0]) / m.config.dt,
                                   [0.01, 0, 0], atol=5e-4)


if __name__ == "__main__":
    unittest.main()
