"""MuJoCo bridge for the reduced B2-Z1 model.

The existing OCP uses Pinocchio ordering while the reference RTWholeBodyMPPI
model uses MuJoCo traversal/actuator ordering.  This module keeps that
difference out of both controllers and provides the common simulation plant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import mujoco
import numpy as np


PIN_JOINT_NAMES: Final = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "joint1", "joint2", "joint3", "joint4",
)

ACTUATOR_NAMES: Final = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "joint1", "joint2", "joint3", "joint4",
)

FOOT_NAMES: Final = ("FR", "FL", "RR", "RL")


def _actuator_name(joint_name: str) -> str:
    if joint_name.startswith("joint"):
        return joint_name
    return joint_name.removesuffix("_joint")


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Return the active rotation matrix for a MuJoCo [w, x, y, z] quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


class B2Z1MujocoEnv:
    """Shared MuJoCo plant with explicit Pinocchio/MuJoCo state conversion."""

    def __init__(
        self,
        model_path: str | Path = "robots/mppi_models/scene.xml",
        timestep: float | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if timestep is not None:
            self.model.opt.timestep = float(timestep)
        self.data = mujoco.MjData(self.model)

        if (self.model.nq, self.model.nv, self.model.nu) != (23, 22, 16):
            raise ValueError(
                "Expected reduced B2-Z1 dimensions (23, 22, 16), got "
                f"({self.model.nq}, {self.model.nv}, {self.model.nu})"
            )

        actual_actuators = tuple(
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            for actuator_id in range(self.model.nu)
        )
        if actual_actuators != ACTUATOR_NAMES:
            raise ValueError(
                f"Unexpected actuator order: {actual_actuators}; "
                f"expected {ACTUATOR_NAMES}"
            )

        self.pin_to_actuator = np.array([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                _actuator_name(joint_name),
            )
            for joint_name in PIN_JOINT_NAMES
        ])
        self.pin_to_mj_qpos = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
            ]
            for joint_name in PIN_JOINT_NAMES
        ])
        self.pin_to_mj_qvel = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
            ]
            for joint_name in PIN_JOINT_NAMES
        ])

        self.kp = np.asarray(self.model.actuator_gainprm[:, 0], dtype=float)
        self.kd = -np.asarray(self.model.actuator_biasprm[:, 2], dtype=float)
        if np.any(self.kp <= 0.0) or np.any(self.kd < 0.0):
            raise ValueError("Expected affine position-PD actuators")

        self.action_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.action_max = self.model.actuator_ctrlrange[:, 1].copy()

    def reset(self, keyframe: str = "stand") -> np.ndarray:
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe
        )
        if key_id < 0:
            raise ValueError(f"Unknown MuJoCo keyframe: {keyframe}")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel)).copy()

    def pin_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mj_to_pin_state(self.data.qpos, self.data.qvel)

    def set_pin_state(self, q_pin: np.ndarray, v_pin: np.ndarray) -> None:
        qpos, qvel = self.pin_to_mj_state(q_pin, v_pin)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def pin_to_mj_state(
        self, q_pin: np.ndarray, v_pin: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        q_pin = np.asarray(q_pin, dtype=float).reshape(-1)
        v_pin = np.asarray(v_pin, dtype=float).reshape(-1)
        if q_pin.shape != (23,) or v_pin.shape != (22,):
            raise ValueError(
                f"Expected Pinocchio q/v shapes (23,)/(22,), got "
                f"{q_pin.shape}/{v_pin.shape}"
            )

        qpos = np.zeros(self.model.nq)
        qvel = np.zeros(self.model.nv)
        qpos[:3] = q_pin[:3]
        qpos[3:7] = q_pin[[6, 3, 4, 5]]
        qpos[self.pin_to_mj_qpos] = q_pin[7:]

        rotation = quat_wxyz_to_matrix(qpos[3:7])
        qvel[:3] = rotation @ v_pin[:3]
        qvel[3:6] = v_pin[3:6]
        qvel[self.pin_to_mj_qvel] = v_pin[6:]
        return qpos, qvel

    def mj_to_pin_state(
        self, qpos: np.ndarray, qvel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        qvel = np.asarray(qvel, dtype=float).reshape(-1)
        if qpos.shape != (23,) or qvel.shape != (22,):
            raise ValueError(
                f"Expected MuJoCo qpos/qvel shapes (23,)/(22,), got "
                f"{qpos.shape}/{qvel.shape}"
            )

        q_pin = np.zeros(23)
        v_pin = np.zeros(22)
        q_pin[:3] = qpos[:3]
        q_pin[3:7] = qpos[[4, 5, 6, 3]]
        q_pin[7:] = qpos[self.pin_to_mj_qpos]

        rotation = quat_wxyz_to_matrix(qpos[3:7])
        v_pin[:3] = rotation.T @ qvel[:3]
        v_pin[3:6] = qvel[3:6]
        v_pin[6:] = qvel[self.pin_to_mj_qvel]
        return q_pin, v_pin

    def pin_joint_to_action(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != (16,):
            raise ValueError(f"Expected 16 joint values, got {values.shape}")
        action = np.empty(16)
        action[self.pin_to_actuator] = values
        return action

    def action_to_pin_joint(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape != (16,):
            raise ValueError(f"Expected 16 actuator values, got {values.shape}")
        return values[self.pin_to_actuator]

    def mpc_reference_to_action(
        self,
        q_ref_pin: np.ndarray,
        v_ref_pin: np.ndarray,
        tau_ff_pin: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode MPC PD+feedforward as the reference affine-PD position action.

        For the MJCF actuator
            tau = kp * (ctrl - q) - kd * qdot,
        choosing
            ctrl = q_ref + (tau_ff + kd * v_ref) / kp
        exactly recovers the MPC command before ctrl/force saturation.
        """
        q_ref = self.pin_joint_to_action(q_ref_pin)
        v_ref = self.pin_joint_to_action(v_ref_pin)
        tau_ff = self.pin_joint_to_action(tau_ff_pin)
        raw_action = q_ref + (tau_ff + self.kd * v_ref) / self.kp
        clipped_action = np.clip(raw_action, self.action_min, self.action_max)
        return clipped_action, raw_action

    def step(self, action: np.ndarray, n_substeps: int = 1) -> np.ndarray:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape != (16,):
            raise ValueError(f"Expected action shape (16,), got {action.shape}")
        self.data.ctrl[:] = np.clip(
            action, self.action_min, self.action_max
        )
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self.observation()

    def actuator_force_pin(self) -> np.ndarray:
        return self.action_to_pin_joint(self.data.actuator_force)

    def sensor(self, name: str) -> np.ndarray:
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise ValueError(f"Unknown sensor: {name}")
        address = self.model.sensor_adr[sensor_id]
        dimension = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[address:address + dimension].copy()

    def foot_contacts(self) -> np.ndarray:
        return np.array([
            self.sensor(f"{foot}_touch")[0] > 0.0 for foot in FOOT_NAMES
        ])

    def close(self) -> None:
        """Compatibility hook for comparison runners."""
