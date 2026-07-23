"""Regression tests for the non-invasive MuJoCo comparison path."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from optimization.mppi_mujoco import (
    MPPIConfig,
    MPCTrackingReference,
    MujocoMPPI,
)
from utils.mujoco_b2z1 import B2Z1MujocoEnv


class MujocoBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env = B2Z1MujocoEnv(timestep=0.015)
        self.env.reset("stand")

    def tearDown(self) -> None:
        self.env.close()

    def test_model_contract_and_task_sensors(self) -> None:
        self.assertEqual(
            (self.env.model.nq, self.env.model.nv, self.env.model.nu),
            (23, 22, 16),
        )
        for name in (
            "base_linvel", "ee_linvel",
            "FR_linvel", "FL_linvel", "RR_linvel", "RL_linvel",
        ):
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    self.env.model, mujoco.mjtObj.mjOBJ_SENSOR, name
                ),
                0,
            )

    def test_pin_mujoco_state_round_trip(self) -> None:
        q_pin, v_pin = self.env.pin_state()
        quaternion = np.array([0.2, -0.1, 0.3, 0.9])
        q_pin[3:7] = quaternion / np.linalg.norm(quaternion)
        v_pin[:] = np.linspace(-0.4, 0.5, len(v_pin))
        qpos, qvel = self.env.pin_to_mj_state(q_pin, v_pin)
        q_back, v_back = self.env.mj_to_pin_state(qpos, qvel)
        np.testing.assert_allclose(q_back, q_pin, atol=1e-12)
        np.testing.assert_allclose(v_back, v_pin, atol=1e-12)

    def test_mpc_equivalent_position_command(self) -> None:
        q_pin, v_pin = self.env.pin_state()
        q_ref = q_pin[7:] + 1.0e-3
        v_ref = np.linspace(-0.02, 0.02, 16)
        tau_ff = np.linspace(-0.5, 0.5, 16)
        action, raw_action = self.env.mpc_reference_to_action(
            q_ref, v_ref, tau_ff
        )
        np.testing.assert_allclose(action, raw_action, atol=0.0)

        self.env.data.ctrl[:] = action
        mujoco.mj_forward(self.env.model, self.env.data)
        q_actual = self.env.data.qpos[self.env.pin_to_mj_qpos]
        v_actual = self.env.data.qvel[self.env.pin_to_mj_qvel]
        q_ref_action = self.env.pin_joint_to_action(q_ref)
        v_ref_action = self.env.pin_joint_to_action(v_ref)
        tau_ff_action = self.env.pin_joint_to_action(tau_ff)
        expected = (
            tau_ff_action
            + self.env.kp * (
                q_ref_action
                - self.env.data.qpos[
                    self.env.model.jnt_qposadr[
                        self.env.model.actuator_trnid[:, 0]
                    ]
                ]
            )
            + self.env.kd * (
                v_ref_action
                - self.env.data.qvel[
                    self.env.model.jnt_dofadr[
                        self.env.model.actuator_trnid[:, 0]
                    ]
                ]
            )
        )
        np.testing.assert_allclose(
            self.env.data.actuator_force, expected, atol=1e-10
        )


class MujocoMPPITest(unittest.TestCase):
    def test_ee_reference_is_frozen_at_update_state(self) -> None:
        env = B2Z1MujocoEnv(timestep=0.015)
        observation = env.reset("stand")
        reference = MPCTrackingReference(
            base_vel_des=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2]),
            arm_vel_des=np.array([0.1, 0.0, -0.2]),
        )
        controller = MujocoMPPI(
            config=MPPIConfig(
                horizon=5,
                n_samples=3,
                n_workers=1,
                n_knots=3,
            ),
            reference=reference,
        )
        try:
            rotation = np.asarray(env.data.xmat[
                mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
                )
            ]).reshape(3, 3)
            expected = rotation @ reference.arm_vel_des
            expected[2] = reference.arm_vel_des[2]
            expected += reference.base_vel_des[:3]
            expected += np.cross(
                reference.base_vel_des[3:],
                env.sensor("ee_pos") - env.sensor("base_pos"),
            )

            captured = {}
            original_cost = controller._cost

            def capture_cost(*args):
                captured["joint_position_reference"] = args[-2].copy()
                captured["desired_ee_velocity"] = args[-1].copy()
                return original_cost(*args)

            controller._cost = capture_cost
            controller.update(observation, 0.0)
            np.testing.assert_allclose(
                captured["desired_ee_velocity"], expected, atol=1e-12
            )
            np.testing.assert_allclose(
                controller.debug_info["desired_ee_velocity"],
                expected,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                captured["joint_position_reference"],
                controller.last_nominal,
                atol=1e-12,
            )
            self.assertFalse(np.allclose(
                captured["joint_position_reference"],
                controller.stand_action[None, :],
            ))
        finally:
            controller.close()
            env.close()

    def test_short_rollout_update(self) -> None:
        env = B2Z1MujocoEnv(timestep=0.015)
        observation = env.reset("stand")
        controller = MujocoMPPI(
            config=MPPIConfig(
                horizon=5,
                n_samples=3,
                n_workers=1,
                n_knots=3,
            ),
            reference=MPCTrackingReference(),
        )
        try:
            action = controller.update(observation, 0.0)
            self.assertEqual(action.shape, (16,))
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(np.isfinite(controller.debug_info["cost_min"]))
        finally:
            controller.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
