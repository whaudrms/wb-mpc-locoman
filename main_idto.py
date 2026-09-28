"""Headless B2 + Z1 IDTO optimization and closed-loop Drake simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.systems.analysis import Simulator

from idto_mpc import B2Z1MPC, MPCConfig, StateAdapter, build_model
from idto_mpc.model import joint_names


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("solve", "simulate"), default="simulate")
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--arm-joints", type=int, default=4, choices=range(7))
    parser.add_argument("--nodes", type=int, default=14)
    parser.add_argument("--dt", type=float, default=0.04, help="fixed prediction step, seconds")
    parser.add_argument("--mpc-period", type=float, default=0.04)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--initial-iterations", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vx", type=float, default=0.)
    parser.add_argument("--vy", type=float, default=0.)
    parser.add_argument("--yaw-rate", type=float, default=0.)
    parser.add_argument("--arm-velocity", nargs=3, type=float, default=[0., 0., 0.],
                        metavar=("VX", "VY", "VZ"), help="relative EE velocity in body frame")
    parser.add_argument("--meshcat", action="store_true")
    parser.add_argument("--output", type=Path, help="write trajectory/telemetry to an npz file")
    args = parser.parse_args()
    for name in ("duration", "mpc_period", "sim_dt"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.mpc_period > args.nodes * args.dt:
        parser.error("MPC period must not exceed the prediction horizon")
    if not np.isclose(args.mpc_period / args.sim_dt, round(args.mpc_period / args.sim_dt)):
        parser.error("MPC period must be an integer multiple of simulation dt")
    return args


def run(args):
    cfg = MPCConfig(arm_joints=args.arm_joints, nodes=args.nodes, dt=args.dt,
                    iterations=args.iterations, initial_iterations=args.initial_iterations,
                    threads=args.threads)
    controller = B2Z1MPC(cfg)
    print(json.dumps({"nq": controller.plant.num_positions(),
                      "nv": controller.plant.num_velocities(),
                      "actuators": controller.plant.num_actuators(),
                      "q_pin_initial": controller.q0_pin.tolist()}, sort_keys=True))
    base = np.array([args.vx, args.vy, 0., 0., 0., args.yaw_rate])
    diag = controller.solve(controller.q0_pin, controller.v0_pin, 0., base, args.arm_velocity)
    print(json.dumps({"time": 0., **diag}, sort_keys=True))
    if args.mode == "solve":
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.output, q=controller.solution[0], v=controller.solution[1],
                     tau=controller.solution[2], joint_names=controller.adapter.names)
        return

    builder, plant, scene_graph, _ = build_model(args.sim_dt, args.arm_joints, simulation=True)
    meshcat, visualizer = None, None
    if args.meshcat:
        from pydrake.geometry import MeshcatVisualizer, StartMeshcat
        # The geometry source is already part of this builder.
        meshcat = StartMeshcat()
        visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    adapter = StateAdapter(plant, joint_names(args.arm_joints))
    q, v = adapter.to_drake(controller.q0_pin, controller.v0_pin)
    plant.SetPositionsAndVelocities(plant_context, np.r_[q, v])
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(len(adapter.names)))
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    if visualizer:
        visualizer.StartRecording()

    # Feedforward + joint PD runs at simulation frequency, MPC at mpc_period.
    kp = np.array([100. if n.endswith("_joint") else 40. for n in adapter.names])
    kd = np.array([5. if n.endswith("_joint") else 3. for n in adapter.names])
    samples, positions, velocities, efforts, timings, residuals = [], [], [], [], [], []
    period_steps = round(args.mpc_period / args.sim_dt)
    clipped_steps = 0
    for step in range(int(np.ceil(args.duration / args.sim_dt))):
        t = context.get_time()
        q = plant.GetPositions(plant_context).copy()
        v = plant.GetVelocities(plant_context).copy()
        qp, vp = adapter.to_pin(q, v)
        if not np.isfinite(np.r_[qp, vp]).all() or qp[2] < 0.25:
            raise RuntimeError(f"Simulation lost valid standing state at t={t:.3f}")
        if step and step % period_steps == 0:
            diag = controller.solve(qp, vp, t, base, args.arm_velocity)
            print(json.dumps({"time": round(t, 4), **diag}, sort_keys=True), flush=True)
        if step % period_steps == 0:
            timings.append(diag["solve_ms"])
            residuals.append(diag["base_residual_max"])
        q_des, v_des, tau_ff = controller.command(t)
        tau = tau_ff + kp * (q_des - qp[7:]) + kd * (v_des - vp[6:])
        clipped_steps += int(np.any(np.abs(tau) > adapter.effort_limits))
        tau = np.clip(tau, -adapter.effort_limits, adapter.effort_limits)
        plant.get_actuation_input_port().FixValue(plant_context, adapter.actuator_torques(tau))
        samples.append(t)
        positions.append(qp)
        velocities.append(vp)
        efforts.append(tau)
        simulator.AdvanceTo(min((step + 1) * args.sim_dt, args.duration))

    q_end, v_end = adapter.to_pin(plant.GetPositions(plant_context), plant.GetVelocities(plant_context))
    if not np.isfinite(np.r_[q_end, v_end]).all() or q_end[2] < 0.25:
        raise RuntimeError("Simulation ended outside the valid standing state")
    samples.append(context.get_time())
    positions.append(q_end)
    velocities.append(v_end)
    # Effort samples describe intervals; state samples include the final endpoint.
    from pydrake.math import RollPitchYaw
    from pydrake.common.eigen_geometry import Quaternion
    tilts = [RollPitchYaw(Quaternion(p[[6, 3, 4, 5]])).vector()[:2] for p in positions]
    summary = {"simulated_seconds": context.get_time(), "final_base_xyz": q_end[:3].tolist(),
               "minimum_base_height": float(min(p[2] for p in positions)),
               "max_abs_joint_velocity": float(np.max(np.abs(np.asarray(velocities)[:, 6:]))),
               "max_abs_roll_pitch": float(np.max(np.abs(tilts))),
               "saturated_control_steps": clipped_steps,
               "mean_solve_ms": float(np.mean(timings)),
               "max_solve_ms": float(np.max(timings)),
               "max_base_residual": float(np.max(residuals)),
               "mpc_period_ms": args.mpc_period * 1000,
               "deadline_misses": int(np.count_nonzero(np.asarray(timings) > args.mpc_period * 1000)),
               "realtime_verified": False}
    print(json.dumps(summary, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output, time=samples, q_pin=positions, v_pin=velocities,
                 tau_pin=efforts, solve_ms=timings, base_residual=residuals,
                 q_pin_final=q_end, v_pin_final=v_end, joint_names=adapter.names,
                 summary=json.dumps(summary))
    if visualizer:
        visualizer.StopRecording()
        visualizer.PublishRecording()
        print(f"Playback: {meshcat.web_url()}")
        input("Press Enter to close Meshcat. ")


if __name__ == "__main__":
    run(parse_args())
