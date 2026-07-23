"""Closed-loop WB-MPC vs MuJoCo-MPPI comparison on one simulation plant.

This is intentionally separate from ``main.py`` so the original solver example
and its predicted-state MPC loop remain unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio as pin

from optimization.mppi_mujoco import (
    MPPIConfig,
    MPCTrackingReference,
    MujocoMPPI,
)
from optimization.mujoco_mpc_adapter import WholeBodyMPCAdapter
from utils.gait_sequence import GaitSequence
from utils.mujoco_b2z1 import B2Z1MujocoEnv, quat_wxyz_to_matrix
from utils.robot import B2_Z1


def ocp_horizon_time(
    nodes: int, dt_min: float, dt_max: float
) -> float:
    gamma = (dt_max / dt_min) ** (1.0 / (nodes - 1))
    return float(sum(dt_min * gamma**index for index in range(nodes)))


def desired_ee_velocity(
    env: B2Z1MujocoEnv,
    reference: MPCTrackingReference,
) -> np.ndarray:
    rotation = quat_wxyz_to_matrix(env.data.qpos[3:7])
    velocity = rotation @ reference.arm_vel_des
    velocity[2] = reference.arm_vel_des[2]
    velocity += reference.base_vel_des[:3]
    velocity += np.cross(
        reference.base_vel_des[3:],
        env.sensor("ee_pos") - env.sensor("base_pos"),
    )
    return velocity


def run_episode(
    controller_name: str,
    args: argparse.Namespace,
    reference: MPCTrackingReference,
) -> dict[str, np.ndarray]:
    env = B2Z1MujocoEnv(args.scene, timestep=args.dt)
    observation = env.reset("stand")
    gait = GaitSequence(reference.gait_type, reference.gait_period)

    controller = None
    if controller_name == "mppi":
        defaults = MPPIConfig()
        joint_position_weight = np.asarray(
            defaults.joint_position_weight
        )
        joint_position_weight *= args.joint_position_weight_scale
        joint_position_weight[-4:] *= args.arm_posture_weight_scale
        joint_velocity_weight = np.asarray(
            defaults.joint_velocity_weight
        )
        joint_velocity_weight[-4:] *= args.arm_velocity_weight_scale
        noise_sigma = args.noise_scale * np.asarray(
            defaults.noise_sigma
        )
        noise_sigma[-4:] *= args.arm_noise_scale
        controller = MujocoMPPI(
            model_path=args.rollout_model,
            config=MPPIConfig(
                timestep=args.dt,
                horizon=args.mppi_horizon,
                n_samples=args.samples,
                n_workers=args.workers,
                temperature=args.temperature,
                seed=args.seed,
                noise_sigma=tuple(noise_sigma),
                joint_position_weight=tuple(joint_position_weight),
                joint_velocity_weight=tuple(joint_velocity_weight),
                base_velocity_weight=tuple(
                    args.base_velocity_weight_scale
                    * np.asarray(defaults.base_velocity_weight)
                ),
                ee_velocity_weight=args.ee_velocity_weight,
            ),
            reference=reference,
        )
        base_height_reference = controller.config.base_height_target
    elif controller_name == "mpc":
        robot = B2_Z1(
            reference_pose="standing_with_arm_up", arm_joints=4
        )
        robot.set_gait_sequence(
            reference.gait_type, reference.gait_period
        )
        pin.computeAllTerms(
            robot.model, robot.data, robot.q0, np.zeros(robot.nv)
        )
        controller = WholeBodyMPCAdapter(
            robot,
            solver=args.solver,
            nodes=args.nodes,
            tau_nodes=args.tau_nodes,
            dt_min=args.dt,
            dt_max=args.dt_max,
            swing_height=reference.swing_height,
            swing_vel_limits=reference.swing_vel_limits,
            base_vel_des=reference.base_vel_des,
            arm_vel_des=reference.arm_vel_des,
            arm_force_des=reference.arm_force_des,
        )
        base_height_reference = float(robot.q0[2])
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    log: dict[str, list] = {
        "time": [],
        "qpos": [],
        "qvel": [],
        "action": [],
        "torque": [],
        "solve_time": [],
        "base_velocity": [],
        "base_velocity_reference": [],
        "base_velocity_error": [],
        "ee_velocity": [],
        "ee_velocity_reference": [],
        "ee_velocity_error": [],
        "contact_mismatch": [],
        "base_height": [],
        "base_height_reference": [],
        "command_clip_fraction": [],
        "constraint_violation": [],
    }

    try:
        for step in range(args.steps):
            t_current = step * args.dt
            if controller_name == "mppi":
                action = controller.update(observation, t_current)
                solve_time = controller.solve_time
                command_clip_fraction = 0.0
                constraint_violation = np.nan
            else:
                q_pin, v_pin = env.pin_state()
                result = controller.step(q_pin, v_pin, t_current)
                action, raw_action = env.mpc_reference_to_action(
                    result.q_ref, result.v_ref, result.tau_ff
                )
                solve_time = result.solve_time
                command_clip_fraction = float(np.mean(
                    np.abs(action - raw_action) > 1e-10
                ))
                constraint_violation = result.constraint_violation

            observation = env.step(action)
            q_pin, v_pin = env.pin_state()
            actual_ee_velocity = env.sensor("ee_linvel")
            target_ee_velocity = desired_ee_velocity(env, reference)
            contact_schedule, _ = gait.get_gait_schedule(
                t_current + args.dt, [args.dt], 1
            )
            contact_mismatch = np.mean(
                env.foot_contacts() != contact_schedule[:, 0].astype(bool)
            )

            log["time"].append(t_current + args.dt)
            log["qpos"].append(env.data.qpos.copy())
            log["qvel"].append(env.data.qvel.copy())
            log["action"].append(action.copy())
            log["torque"].append(env.actuator_force_pin())
            log["solve_time"].append(solve_time)
            log["base_velocity"].append(v_pin[:6].copy())
            log["base_velocity_reference"].append(
                reference.base_vel_des.copy()
            )
            log["base_velocity_error"].append(
                np.linalg.norm(v_pin[:6] - reference.base_vel_des)
            )
            log["ee_velocity"].append(actual_ee_velocity.copy())
            log["ee_velocity_reference"].append(
                target_ee_velocity.copy()
            )
            log["ee_velocity_error"].append(
                np.linalg.norm(
                    actual_ee_velocity - target_ee_velocity
                )
            )
            log["contact_mismatch"].append(contact_mismatch)
            log["base_height"].append(q_pin[2])
            log["base_height_reference"].append(
                base_height_reference
            )
            log["command_clip_fraction"].append(
                command_clip_fraction
            )
            log["constraint_violation"].append(
                constraint_violation
            )
    finally:
        if hasattr(controller, "close"):
            controller.close()
        env.close()

    return {
        key: np.asarray(values) for key, values in log.items()
    }


def summarize(name: str, result: dict[str, np.ndarray]) -> None:
    solve_ms = 1.0e3 * result["solve_time"]
    print(f"\n{name.upper()} ({len(solve_ms)} closed-loop steps)")
    print(
        f"  solve time: mean={np.mean(solve_ms):.2f} ms, "
        f"p95={np.percentile(solve_ms, 95):.2f} ms"
    )
    print(
        "  base velocity RMSE norm: "
        f"{np.sqrt(np.mean(result['base_velocity_error'] ** 2)):.4f}"
    )
    print(
        "  EE velocity RMSE norm: "
        f"{np.sqrt(np.mean(result['ee_velocity_error'] ** 2)):.4f}"
    )
    print(
        "  contact mismatch: "
        f"{100.0 * np.mean(result['contact_mismatch']):.1f}%"
    )
    print(
        f"  minimum base height: {np.min(result['base_height']):.3f} m"
    )
    if name == "mpc":
        print(
            "  equivalent-command clipped joints: "
            f"{100.0 * np.mean(result['command_clip_fraction']):.1f}%"
        )
        print(
            "  max OCP constraint violation: "
            f"{np.nanmax(result['constraint_violation']):.3e}"
        )


def plot_results(
    results: dict[str, dict[str, np.ndarray]], output: Path
) -> None:
    """Save controller tracking, effort, contact, and timing plots."""
    figure, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    series = (
        ("base_velocity_error", "Base velocity error norm", "m/s"),
        ("ee_velocity_error", "EE velocity error norm", "m/s"),
        ("torque", "Joint torque norm", "Nm"),
        ("base_height", "Base height", "m"),
        ("contact_mismatch", "Contact schedule mismatch", "fraction"),
        ("solve_time", "Controller solve time", "ms"),
    )
    for axis, (key, title, unit) in zip(axes.flat, series):
        for name, result in results.items():
            values = result[key]
            if key == "torque":
                values = np.linalg.norm(values, axis=1)
            elif key == "solve_time":
                values = 1.0e3 * values
            line, = axis.plot(
                result["time"], values, label=name.upper()
            )
            if key == "base_height":
                axis.plot(
                    result["time"],
                    result["base_height_reference"],
                    linestyle="--",
                    color=line.get_color(),
                    label=f"{name.upper()} reference",
                )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("time [s]")
    axes[0, 0].legend()
    axes[1, 1].legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved plot: {output}")

    tracking_output = output.with_name(
        f"{output.stem}_tracking{output.suffix}"
    )
    tracking_figure, tracking_axes = plt.subplots(
        5, 2, figsize=(13, 15), sharex=True
    )
    tracking_series = (
        ("base_velocity", "base_velocity_reference", 0, "Base linear vx", "m/s"),
        ("base_velocity", "base_velocity_reference", 1, "Base linear vy", "m/s"),
        ("base_velocity", "base_velocity_reference", 2, "Base linear vz", "m/s"),
        ("base_velocity", "base_velocity_reference", 3, "Base angular wx", "rad/s"),
        ("base_velocity", "base_velocity_reference", 4, "Base angular wy", "rad/s"),
        ("base_velocity", "base_velocity_reference", 5, "Base angular wz", "rad/s"),
        ("ee_velocity", "ee_velocity_reference", 0, "EE velocity vx", "m/s"),
        ("ee_velocity", "ee_velocity_reference", 1, "EE velocity vy", "m/s"),
        ("ee_velocity", "ee_velocity_reference", 2, "EE velocity vz", "m/s"),
        ("base_height", "base_height_reference", None, "Base height", "m"),
    )
    for axis, (
        value_key,
        reference_key,
        component,
        title,
        unit,
    ) in zip(tracking_axes.flat, tracking_series):
        for name, result in results.items():
            values = result[value_key]
            references = result[reference_key]
            if component is not None:
                values = values[:, component]
                references = references[:, component]
            line, = axis.plot(
                result["time"],
                values,
                label=f"{name.upper()} actual",
            )
            axis.plot(
                result["time"],
                references,
                linestyle="--",
                color=line.get_color(),
                label=f"{name.upper()} reference",
            )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(True, alpha=0.3)
    for axis in tracking_axes[-1]:
        axis.set_xlabel("time [s]")
    handles, labels = tracking_axes[0, 0].get_legend_handles_labels()
    tracking_figure.legend(
        handles, labels, loc="upper center", ncol=max(2, len(handles))
    )
    tracking_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    tracking_figure.savefig(tracking_output, dpi=160)
    plt.close(tracking_figure)
    print(f"Saved raw tracking plot: {tracking_output}")


class _TrajectoryRenderer:
    def __init__(
        self, scene: str | Path, width: int, height: int
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(
            str(Path(scene).resolve())
        )
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(
            self.model, height=height, width=width
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 2.4
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0

    def render(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = self.data.qpos[:3]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


def replay_results(
    results: dict[str, dict[str, np.ndarray]],
    scene: str | Path,
    dt: float,
    width: int,
    height: int,
    replay_speed: float,
    render_every: int,
) -> None:
    """Replay controller trajectories in synchronized Matplotlib panels."""
    names = tuple(results)
    renderers = {
        name: _TrajectoryRenderer(scene, width, height)
        for name in names
    }
    figure, axes = plt.subplots(1, len(names), squeeze=False)
    axes = axes[0]
    images = {}
    try:
        for axis, name in zip(axes, names):
            frame = renderers[name].render(
                results[name]["qpos"][0], results[name]["qvel"][0]
            )
            images[name] = axis.imshow(frame)
            axis.set_title(name.upper())
            axis.axis("off")
        figure.tight_layout()
        plt.show(block=False)

        frame_count = min(len(result["time"]) for result in results.values())
        for frame_index in range(0, frame_count, render_every):
            for name in names:
                frame = renderers[name].render(
                    results[name]["qpos"][frame_index],
                    results[name]["qvel"][frame_index],
                )
                images[name].set_data(frame)
            figure.suptitle(
                f"t = {results[names[0]]['time'][frame_index]:.3f} s"
            )
            figure.canvas.draw_idle()
            plt.pause(max(
                dt * render_every / replay_speed, 1.0e-3
            ))
        plt.show()
    finally:
        for renderer in renderers.values():
            renderer.close()


def record_results(
    results: dict[str, dict[str, np.ndarray]],
    scene: str | Path,
    output: Path,
    dt: float,
    width: int,
    height: int,
    replay_speed: float,
    render_every: int,
) -> None:
    """Render a synchronized side-by-side GIF without affecting solve timing."""
    from PIL import Image, ImageDraw

    names = tuple(results)
    renderers = {
        name: _TrajectoryRenderer(scene, width, height)
        for name in names
    }
    frames = []
    frame_count = min(len(result["time"]) for result in results.values())
    try:
        for frame_index in range(0, frame_count, render_every):
            rendered = []
            for name in names:
                frame = renderers[name].render(
                    results[name]["qpos"][frame_index],
                    results[name]["qvel"][frame_index],
                )
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 150, 28), fill=(0, 0, 0))
                draw.text((8, 7), name.upper(), fill=(255, 255, 255))
                rendered.append(np.asarray(image))
            combined = np.concatenate(rendered, axis=1)
            image = Image.fromarray(combined)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, height - 28, 190, height), fill=(0, 0, 0))
            draw.text(
                (8, height - 21),
                f"t={results[names[0]]['time'][frame_index]:.3f}s",
                fill=(255, 255, 255),
            )
            frames.append(image)
    finally:
        for renderer in renderers.values():
            renderer.close()

    if not frames:
        raise RuntimeError("No trajectory frames were available to record")
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(
        1, int(1.0e3 * dt * render_every / replay_speed)
    )
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved replay: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare WB-MPC and position-action MuJoCo MPPI"
    )
    parser.add_argument(
        "--controller", choices=("mpc", "mppi", "both"), default="both"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--nodes", type=int, default=14)
    parser.add_argument("--tau-nodes", type=int, default=3)
    parser.add_argument("--dt-max", type=float, default=0.08)
    parser.add_argument(
        "--samples", type=int, default=MPPIConfig().n_samples
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--arm-noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--ee-velocity-weight", type=float,
        default=MPPIConfig().ee_velocity_weight,
    )
    parser.add_argument(
        "--base-velocity-weight-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--joint-position-weight-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--arm-posture-weight-scale", type=float, default=1.0,
    )
    parser.add_argument(
        "--arm-velocity-weight-scale", type=float, default=1.0,
    )
    parser.add_argument("--solver", choices=("fatrop", "ipopt"), default="fatrop")
    parser.add_argument(
        "--base-vel", nargs=6, type=float,
        default=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    parser.add_argument(
        "--arm-vel", nargs=3, type=float,
        default=(0.1, 0.0, -0.2),
    )
    parser.add_argument(
        "--gait", choices=("trot", "walk", "stand"), default="trot"
    )
    parser.add_argument("--gait-period", type=float, default=0.8)
    parser.add_argument("--swing-height", type=float, default=0.07)
    parser.add_argument(
        "--swing-vel-limits", nargs=2, type=float, default=(0.1, -0.2)
    )
    parser.add_argument(
        "--scene", default="robots/mppi_models/scene.xml"
    )
    parser.add_argument(
        "--rollout-model",
        default="robots/mppi_models/b2_z1_base.xml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/mujoco_mpc_mppi.npz"),
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save tracking, torque, contact, and timing plots.",
    )
    parser.add_argument(
        "--plot-output", type=Path,
        default=Path("results/mujoco_mpc_mppi.png"),
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Replay completed trajectories in synchronized viewer panels.",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Save a synchronized side-by-side MuJoCo GIF.",
    )
    parser.add_argument(
        "--record-output", type=Path,
        default=Path("results/mujoco_mpc_mppi.gif"),
    )
    parser.add_argument("--render-width", type=int, default=480)
    parser.add_argument("--render-height", type=int, default=360)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--replay-speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.render_width < 1 or args.render_height < 1:
        parser.error("render dimensions must be positive")
    if args.render_every < 1 or args.replay_speed <= 0.0:
        parser.error("--render-every and --replay-speed must be positive")
    tuning_values = (
        args.noise_scale,
        args.arm_noise_scale,
        args.ee_velocity_weight,
        args.base_velocity_weight_scale,
        args.joint_position_weight_scale,
        args.arm_posture_weight_scale,
        args.arm_velocity_weight_scale,
    )
    if not np.isfinite(tuning_values).all() or np.any(
        np.asarray(tuning_values) < 0.0
    ):
        parser.error("MPPI tuning scales and weights must be non-negative")
    horizon_time = ocp_horizon_time(
        args.nodes, args.dt, args.dt_max
    )
    args.mppi_horizon = max(2, int(round(horizon_time / args.dt)))
    return args


def main() -> None:
    args = parse_args()
    reference = MPCTrackingReference(
        base_vel_des=args.base_vel,
        arm_vel_des=args.arm_vel,
        arm_force_des=np.zeros(3),
        gait_type=args.gait,
        gait_period=args.gait_period,
        swing_height=args.swing_height,
        swing_vel_limits=tuple(args.swing_vel_limits),
    )
    names = (
        ("mpc", "mppi")
        if args.controller == "both"
        else (args.controller,)
    )
    results = {}
    for name in names:
        results[name] = run_episode(name, args, reference)
        summarize(name, results[name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flattened = {
        f"{controller}_{key}": value
        for controller, result in results.items()
        for key, value in result.items()
    }
    np.savez_compressed(args.output, **flattened)
    print(f"\nSaved: {args.output}")
    if args.plot:
        plot_results(results, args.plot_output)
    if args.record:
        record_results(
            results,
            args.scene,
            args.record_output,
            args.dt,
            args.render_width,
            args.render_height,
            args.replay_speed,
            args.render_every,
        )
    if args.visualize:
        replay_results(
            results,
            args.scene,
            args.dt,
            args.render_width,
            args.render_height,
            args.replay_speed,
            args.render_every,
        )


if __name__ == "__main__":
    main()
