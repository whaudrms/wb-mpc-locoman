"""MuJoCo MPPI with the reference controller's position-action core.

The sampling/update flow follows RTWholeBodyMPPI:
  position actions -> affine-PD MuJoCo rollout -> trajectory cost ->
  exponential weighted action update -> receding-horizon shift.

Only the task layer is changed: targets and gait timing are the same
base/arm velocity and contact/swing references consumed by the existing OCP.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from mujoco import rollout
from scipy.interpolate import CubicSpline

from utils.gait_sequence import GaitSequence
from utils.mujoco_b2z1 import FOOT_NAMES, quat_wxyz_to_matrix


@dataclass(frozen=True)
class MPCTrackingReference:
    """Task command shared by WB-MPC and MuJoCo MPPI."""

    base_vel_des: np.ndarray = field(
        default_factory=lambda: np.zeros(6)
    )
    arm_vel_des: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    arm_force_des: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )
    gait_type: str = "trot"
    gait_period: float = 0.8
    swing_height: float = 0.07
    swing_vel_limits: tuple[float, float] = (0.1, -0.2)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "base_vel_des",
            np.asarray(self.base_vel_des, dtype=float).reshape(6),
        )
        object.__setattr__(
            self, "arm_vel_des",
            np.asarray(self.arm_vel_des, dtype=float).reshape(3),
        )
        object.__setattr__(
            self, "arm_force_des",
            np.asarray(self.arm_force_des, dtype=float).reshape(3),
        )


@dataclass(frozen=True)
class MPPIConfig:
    timestep: float = 0.015
    horizon: int = 37
    n_samples: int = 128
    n_workers: int = 5
    temperature: float = 0.1
    sample_type: str = "cubic"
    n_knots: int = 4
    seed: int = 42
    noise_sigma: tuple[float, ...] = (
        0.03, 0.10, 0.10, 0.03, 0.10, 0.10,
        0.03, 0.10, 0.10, 0.03, 0.10, 0.10,
        0.03, 0.03, 0.03, 0.03,
    )
    joint_position_weight: tuple[float, ...] = (
        100.0, 100.0, 100.0,
        100.0, 100.0, 100.0,
        100.0, 100.0, 100.0,
        100.0, 100.0, 100.0,
        100.0, 100.0, 100.0, 100.0,
    )
    joint_velocity_weight: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    )
    base_height_target: float = 0.55
    base_height_weight: float = 0.0
    base_orientation_weight: float = 5_000.0
    base_velocity_weight: tuple[float, ...] = (
        10_000.0, 10_000.0, 10_000.0,
        10_000.0, 10_000.0, 10_000.0,
    )
    torque_weight: tuple[float, ...] = (
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3,
        1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3,
    )
    ee_velocity_weight: float = 75_000.0
    stance_slip_weight: float = 2_000.0
    swing_velocity_weight: float = 2_000.0
    lost_contact_weight: float = 5_000.0
    early_contact_weight: float = 5_000.0
    fall_height: float = 0.30
    fall_penalty: float = 1.0e6
    swing_thigh_lift: float = -0.05
    swing_calf_lift: float = -0.20

    def __post_init__(self) -> None:
        if self.timestep <= 0.0 or self.horizon < 2 or self.n_samples < 1:
            raise ValueError("Invalid MPPI time or batch dimensions")
        if self.temperature <= 0.0:
            raise ValueError("MPPI temperature must be positive")
        vector_sizes = {
            "noise_sigma": (self.noise_sigma, 16),
            "joint_position_weight": (self.joint_position_weight, 16),
            "joint_velocity_weight": (self.joint_velocity_weight, 16),
            "base_velocity_weight": (self.base_velocity_weight, 6),
            "torque_weight": (self.torque_weight, 16),
        }
        for name, (values, expected_size) in vector_sizes.items():
            array = np.asarray(values, dtype=float)
            if array.shape != (expected_size,):
                raise ValueError(
                    f"{name} must contain {expected_size} values"
                )
            if not np.isfinite(array).all() or np.any(array < 0.0):
                raise ValueError(
                    f"{name} values must be finite and non-negative"
                )

        scalar_weights = np.asarray((
            self.base_height_weight,
            self.base_orientation_weight,
            self.ee_velocity_weight,
            self.stance_slip_weight,
            self.swing_velocity_weight,
            self.lost_contact_weight,
            self.early_contact_weight,
            self.fall_penalty,
        ))
        if (
            not np.isfinite(scalar_weights).all()
            or np.any(scalar_weights < 0.0)
        ):
            raise ValueError(
                "MPPI cost weights must be finite and non-negative"
            )
        if not np.isfinite(self.base_height_target):
            raise ValueError("base_height_target must be finite")


class MujocoMPPI:
    """Position-action MPPI with WB-MPC-aligned task references."""

    def __init__(
        self,
        model_path: str | Path = "robots/mppi_models/b2_z1_base.xml",
        config: MPPIConfig | None = None,
        reference: MPCTrackingReference | None = None,
    ) -> None:
        self.config = config or MPPIConfig()
        self.reference = reference or MPCTrackingReference()
        if not np.allclose(self.reference.arm_force_des, 0.0):
            raise NotImplementedError(
                "Nonzero arm_force_des requires an interaction object/wrench "
                "model; the default zero-force MPC task is supported."
            )

        self.model_path = Path(model_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = self.config.timestep
        self.act_dim = self.model.nu
        if (self.model.nq, self.model.nv, self.act_dim) != (23, 22, 16):
            raise ValueError("MujocoMPPI requires the reduced 4-DoF B2-Z1 model")

        self.action_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.action_max = self.model.actuator_ctrlrange[:, 1].copy()
        self.kp = np.asarray(self.model.actuator_gainprm[:, 0], dtype=float)
        self.kd = -np.asarray(self.model.actuator_biasprm[:, 2], dtype=float)
        self.noise_sigma = np.asarray(
            self.config.noise_sigma, dtype=float
        )
        self.joint_position_weight = np.asarray(
            self.config.joint_position_weight, dtype=float
        )
        self.joint_velocity_weight = np.asarray(
            self.config.joint_velocity_weight, dtype=float
        )
        self.base_velocity_weight = np.asarray(
            self.config.base_velocity_weight, dtype=float
        )
        self.torque_weight = np.asarray(
            self.config.torque_weight, dtype=float
        )
        self.rng = np.random.default_rng(self.config.seed)

        self.joint_qpos_indices, self.joint_qvel_indices = (
            self._actuated_joint_indices()
        )
        self.arm_dof_indices = self.joint_qvel_indices[-4:]
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center"
        )
        self.sensor_slices = {
            name: self._sensor_slice(name)
            for name in (
                "base_linvel", "ee_linvel", "base_pos", "ee_pos",
                "FR_linvel", "FL_linvel", "RR_linvel", "RL_linvel",
                "FR_touch", "FL_touch", "RR_touch", "RL_touch",
            )
        }

        stand_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        if stand_id < 0:
            raise ValueError("MuJoCo model is missing the stand keyframe")
        self.stand_action = self.model.key_ctrl[stand_id].copy()

        self.gait = GaitSequence(
            self.reference.gait_type, self.reference.gait_period
        )
        self.correction = np.zeros(
            (self.config.horizon, self.act_dim)
        )
        self.selected_trajectory = np.repeat(
            self.stand_action[None, :], self.config.horizon, axis=0
        )
        self.last_nominal = self.selected_trajectory.copy()
        self.solve_time = 0.0
        self.debug_info: dict[str, float | int | np.ndarray] = {}

        state_dim = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )
        self.state_rollouts = np.empty(
            (self.config.n_samples, self.config.horizon, state_dim)
        )
        self.sensor_rollouts = np.empty(
            (
                self.config.n_samples,
                self.config.horizon,
                self.model.nsensordata,
            )
        )
        self.thread_local = threading.local()
        workers = max(
            1, min(self.config.n_workers, self.config.n_samples)
        )
        self.executor = ThreadPoolExecutor(
            max_workers=workers, initializer=self._thread_initializer
        )
        self.n_workers = workers
        self._closed = False

    def _actuated_joint_indices(self) -> tuple[np.ndarray, np.ndarray]:
        qpos_indices = []
        qvel_indices = []
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            qpos_indices.append(self.model.jnt_qposadr[joint_id])
            qvel_indices.append(self.model.jnt_dofadr[joint_id])
        return np.asarray(qpos_indices), np.asarray(qvel_indices)

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            raise ValueError(
                f"Rollout model is missing required sensor '{name}'"
            )
        start = self.model.sensor_adr[sensor_id]
        return slice(start, start + self.model.sensor_dim[sensor_id])

    def _thread_initializer(self) -> None:
        self.thread_local.data = mujoco.MjData(self.model)

    def _call_rollout(
        self,
        initial_state: np.ndarray,
        actions: np.ndarray,
        states: np.ndarray,
        sensors: np.ndarray,
    ) -> None:
        rollout.rollout(
            self.model,
            self.thread_local.data,
            initial_state=initial_state,
            control=actions,
            state=states,
            sensordata=sensors,
            nroll=states.shape[0],
            nstep=states.shape[1],
            skip_checks=True,
        )

    def _rollout(
        self, observation: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        observation = np.asarray(observation, dtype=float).reshape(-1)
        if observation.shape != (45,):
            raise ValueError(
                f"Expected MuJoCo observation shape (45,), got "
                f"{observation.shape}"
            )

        initial = np.repeat(
            np.concatenate(([0.0], observation))[None, :],
            self.config.n_samples,
            axis=0,
        )
        boundaries = np.linspace(
            0, self.config.n_samples, self.n_workers + 1, dtype=int
        )
        futures = []
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= begin:
                continue
            index = slice(begin, end)
            futures.append(self.executor.submit(
                self._call_rollout,
                initial[index],
                actions[index],
                self.state_rollouts[index],
                self.sensor_rollouts[index],
            ))
        for future in concurrent.futures.as_completed(futures):
            future.result()
        return self.state_rollouts[:, :, 1:], self.sensor_rollouts

    def _sample_actions(self, trajectory: np.ndarray) -> np.ndarray:
        shape = (
            self.config.n_samples,
            self.config.horizon,
            self.act_dim,
        )
        if self.config.sample_type == "normal":
            noise = self.rng.normal(size=shape) * self.noise_sigma
        elif self.config.sample_type == "cubic":
            indices = np.unique(np.rint(np.linspace(
                0, self.config.horizon - 1, self.config.n_knots
            )).astype(int))
            knot_noise = self.rng.normal(
                size=(self.config.n_samples, len(indices), self.act_dim)
            ) * self.noise_sigma
            noise = CubicSpline(indices, knot_noise, axis=1)(
                np.arange(self.config.horizon)
            )
        else:
            raise ValueError(
                f"Unsupported MPPI sample_type: {self.config.sample_type}"
            )
        actions = trajectory[None, :, :] + noise
        actions[0] = trajectory
        return np.clip(actions, self.action_min, self.action_max)

    def _nominal_actions(
        self, observation: np.ndarray, t_current: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dts = [self.config.timestep] * self.config.horizon
        contact, swing = self.gait.get_gait_schedule(
            t_current, dts, self.config.horizon
        )
        nominal = np.repeat(
            self.stand_action[None, :], self.config.horizon, axis=0
        )

        # The gait-conditioned trajectory is both the sampling center and the
        # joint-position cost reference, as in the position-based reference
        # MPPI.  This avoids penalizing every intended swing back toward stand.
        for foot_index in range(4):
            leg_slice = slice(3 * foot_index, 3 * foot_index + 3)
            lift = np.sin(np.pi * swing[foot_index])
            nominal[:, leg_slice.start + 1] += (
                self.config.swing_thigh_lift * lift
            )
            nominal[:, leg_slice.start + 2] += (
                self.config.swing_calf_lift * lift
            )

        # Build a first-order arm-position prior for the MPC arm-velocity task.
        data = mujoco.MjData(self.model)
        data.qpos[:] = observation[:self.model.nq]
        data.qvel[:] = observation[self.model.nq:]
        mujoco.mj_forward(self.model, data)
        base_rotation = quat_wxyz_to_matrix(data.qpos[3:7])
        base_pos = data.sensordata[self.sensor_slices["base_pos"]]
        ee_pos = data.sensordata[self.sensor_slices["ee_pos"]]
        desired_ee_velocity = (
            base_rotation @ self.reference.arm_vel_des
        )
        desired_ee_velocity[2] = self.reference.arm_vel_des[2]
        desired_ee_velocity += self.reference.base_vel_des[:3]
        desired_ee_velocity += np.cross(
            self.reference.base_vel_des[3:],
            ee_pos - base_pos,
        )

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, data, jacp, jacr, self.ee_site_id
        )
        relative_velocity = base_rotation @ self.reference.arm_vel_des
        relative_velocity[2] = self.reference.arm_vel_des[2]
        arm_velocity = np.linalg.pinv(
            jacp[:, self.arm_dof_indices], rcond=1e-5
        ) @ relative_velocity
        arm_q0 = data.qpos[self.joint_qpos_indices[-4:]]
        times = self.config.timestep * (
            np.arange(self.config.horizon) + 1
        )
        nominal[:, -4:] = (
            arm_q0[None, :]
            + times[:, None] * arm_velocity[None, :]
        )
        nominal = np.clip(nominal, self.action_min, self.action_max)
        return nominal, contact.T, swing.T, desired_ee_velocity

    @staticmethod
    def _roll_pitch(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, x, y, z = np.moveaxis(quaternions, -1, 0)
        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = np.arcsin(np.clip(
            2.0 * (w * y - z * x), -1.0, 1.0
        ))
        return roll, pitch

    @staticmethod
    def _world_to_body(
        quaternions: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        w, x, y, z = np.moveaxis(quaternions, -1, 0)
        q_vec = np.stack((x, y, z), axis=-1)
        # Rotate by quaternion conjugate without building one matrix per state.
        return (
            vectors
            - 2.0 * w[..., None] * np.cross(q_vec, vectors)
            + 2.0 * np.cross(q_vec, np.cross(q_vec, vectors))
        )

    @staticmethod
    def _spline_velocity(
        phase: np.ndarray,
        period: float,
        height: float,
        v_liftoff: float,
        v_touchdown: float,
    ) -> np.ndarray:
        """Numpy equivalent of utils.gait_sequence.get_spline_vel_z."""
        phase = np.asarray(phase, dtype=float)
        half = period / 2.0

        def coefficients(p0, v0, p1, v1):
            dpos = p1 - p0
            dvel = v1 - v0
            return (
                v0 * half,
                -(3.0 * v0 + dvel) * half + 3.0 * dpos,
                (2.0 * v0 + dvel) * half - 2.0 * dpos,
            )

        c1_1, c2_1, c3_1 = coefficients(
            0.0, v_liftoff, height, 0.0
        )
        c1_2, c2_2, c3_2 = coefficients(
            height, 0.0, 0.0, v_touchdown
        )
        first = phase < 0.5
        tn = np.where(first, 2.0 * phase, 2.0 * phase - 1.0)
        c1 = np.where(first, c1_1, c1_2)
        c2 = np.where(first, c2_1, c2_2)
        c3 = np.where(first, c3_1, c3_2)
        return (3.0 * c3 * tn * tn + 2.0 * c2 * tn + c1) / half

    def _cost(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        actions: np.ndarray,
        contact_schedule: np.ndarray,
        swing_schedule: np.ndarray,
        joint_position_reference: np.ndarray,
        desired_ee_velocity: np.ndarray,
    ) -> np.ndarray:
        qpos = states[:, :, :self.model.nq]
        qvel = states[:, :, self.model.nq:]
        q_joint = qpos[:, :, self.joint_qpos_indices]
        v_joint = qvel[:, :, self.joint_qvel_indices]

        stage = np.zeros(states.shape[:2])
        q_error = q_joint - joint_position_reference[None, :, :]
        stage += np.sum(
            q_error * q_error * self.joint_position_weight, axis=-1
        )
        stage += np.sum(
            v_joint * v_joint * self.joint_velocity_weight, axis=-1
        )

        base_height_error = (
            qpos[:, :, 2] - self.config.base_height_target
        )
        roll, pitch = self._roll_pitch(qpos[:, :, 3:7])
        stage += (
            self.config.base_height_weight
            * base_height_error * base_height_error
        )
        stage += (
            self.config.base_orientation_weight
            * (roll * roll + pitch * pitch)
        )

        base_velocity = np.empty_like(qvel[:, :, :6])
        base_velocity[:, :, :3] = self._world_to_body(
            qpos[:, :, 3:7], qvel[:, :, :3]
        )
        base_velocity[:, :, 3:] = qvel[:, :, 3:6]
        base_velocity_error = (
            base_velocity - self.reference.base_vel_des
        )
        stage += np.sum(
            base_velocity_error * base_velocity_error
            * self.base_velocity_weight,
            axis=-1,
        )

        # Match the reference controller's actuator-consistent effort cost.
        torque = (
            self.kp * (actions - q_joint) - self.kd * v_joint
        )
        stage += np.sum(
            torque * torque * self.torque_weight, axis=-1
        )

        ee_velocity = sensors[:, :, self.sensor_slices["ee_linvel"]]
        ee_error = ee_velocity - desired_ee_velocity
        stage += self.config.ee_velocity_weight * np.sum(
            ee_error * ee_error, axis=-1
        )

        desired_contact = contact_schedule[None, :, :]
        swing_phase = swing_schedule[None, :, :]
        foot_velocity = np.stack([
            sensors[:, :, self.sensor_slices[f"{foot}_linvel"]]
            for foot in FOOT_NAMES
        ], axis=2)
        touches = np.stack([
            sensors[:, :, self.sensor_slices[f"{foot}_touch"]][..., 0]
            for foot in FOOT_NAMES
        ], axis=2) > 1e-8
        stage += self.config.stance_slip_weight * np.sum(
            desired_contact * np.sum(foot_velocity * foot_velocity, axis=-1),
            axis=-1,
        )
        swing_velocity_des = self._spline_velocity(
            swing_phase,
            self.gait.swing_period,
            self.reference.swing_height,
            self.reference.swing_vel_limits[0],
            self.reference.swing_vel_limits[1],
        )
        swing_error = foot_velocity[:, :, :, 2] - swing_velocity_des
        stage += self.config.swing_velocity_weight * np.sum(
            (1.0 - desired_contact) * swing_error * swing_error,
            axis=-1,
        )
        stage += self.config.lost_contact_weight * np.sum(
            desired_contact * (~touches), axis=-1
        )
        stage += self.config.early_contact_weight * np.sum(
            (1.0 - desired_contact) * touches, axis=-1
        )
        stage += self.config.fall_penalty * (
            qpos[:, :, 2] < self.config.fall_height
        )
        return np.sum(stage, axis=1)

    def update(
        self, observation: np.ndarray, t_current: float
    ) -> np.ndarray:
        start = time.perf_counter()
        nominal, contact, swing, desired_ee_velocity = self._nominal_actions(
            observation, t_current
        )
        trajectory = np.clip(
            nominal + self.correction,
            self.action_min,
            self.action_max,
        )
        actions = self._sample_actions(trajectory)
        states, sensors = self._rollout(observation, actions)
        costs = self._cost(
            states,
            sensors,
            actions,
            contact,
            swing,
            nominal,
            desired_ee_velocity,
        )
        finite = np.isfinite(costs)
        if np.any(finite):
            valid_costs = costs[finite]
            minimum = np.min(valid_costs)
            spread = np.max(valid_costs) - minimum
            weights = np.zeros_like(costs)
            if spread < 1e-12:
                weights[finite] = 1.0
            else:
                weights[finite] = np.exp(
                    -((valid_costs - minimum) / spread)
                    / self.config.temperature
                )
            updated = np.sum(
                weights[:, None, None] * actions, axis=0
            ) / (np.sum(weights) + 1e-12)
            updated = np.clip(
                updated, self.action_min, self.action_max
            )
            best_cost = float(minimum)
        else:
            weights = np.zeros_like(costs)
            updated = trajectory.copy()
            best_cost = float("inf")

        self.selected_trajectory = updated
        selected_correction = updated - nominal
        self.correction[:-1] = selected_correction[1:]
        self.correction[-1] = 0.0
        self.last_nominal = nominal
        self.solve_time = time.perf_counter() - start
        self.debug_info = {
            "cost_min": best_cost,
            "weight_max": float(np.max(weights)),
            "solve_time": self.solve_time,
            "finite_rollouts": int(np.sum(finite)),
            "contact_schedule0": contact[0].copy(),
            "joint_position_reference0": nominal[0].copy(),
            "desired_ee_velocity": desired_ee_velocity.copy(),
        }
        return updated[0].copy()

    def close(self) -> None:
        if not self._closed:
            self.executor.shutdown(wait=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
