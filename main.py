import os

# Small QP blocks benefit from one BLAS thread; set before NumPy/CasADi imports.
blas_threads = 1
os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
os.environ["OMP_NUM_THREADS"] = str(blas_threads)
os.environ["MKL_NUM_THREADS"] = str(blas_threads)

import time
import numpy as np
import pinocchio as pin
import casadi as ca
import matplotlib.pyplot as plt

from args import *
from utils.robot import *
from utils.visualization import visualize_forces
from optimization import make_ocp

# Robot params
robot = B2_Z1(reference_pose="standing_with_arm_up", arm_joints=4)
dynamics = "centroidal_vel"  # see args.py for options

# Tracking targets
base_vel_des = np.array([0.1, 0, 0, 0, 0, 0])  # linear + angular velocity
arm_vel_des = np.array([0.1, 0, -0.2])         # arm EE velocity (relative to the base)
arm_force_des = np.array([0, 0, 0])            # arm EE force (global)

# OCP params
nodes = 14      # OCP nodes
tau_nodes = 3   # centroidal: torque estimates here; RNEA QP: exact limits at all nodes
dt_min = 0.015  # initial time step
dt_max = 0.08   # final time step

# Gait params
gait_type = "trot"              # "trot", "walk" or "stand"
gait_period = 0.8               # seconds
swing_height = 0.07             # meters
swing_vel_limits = [0.1, -0.2]  # meters/second

# Solver
solver = "fatrop"  # "qpOASE" (qpOASES), "OSQP", "HPIPM"; legacy: "qp", "fatrop", "ipopt", "osqp"
qp_condesed = True  # True: condensed QP; False: original QP (HPIPM uses OCP structure)
qp_compile_data = True  # Compile model/Jacobian evaluation once and cache the shared library.
qp_profile = True       # Print QP phase times when log_mode="detailed".
warm_start = True
compile_solver = False
load_compiled_solver = None  # None or <filename> in "codegen/lib/"

# MPC
mpc_loops = 200

# Debug
log_mode = "detailed"  # "original": NLP-style solve/CV logs; "detailed": full MPC + QP phases.
plot = False  # plot joint positions, velocities, torques


def get_solver_configuration():
    """Map main.py choices to the single-QP or legacy nonlinear solver API."""
    if dynamics.endswith("_qp"):
        backends = {"qpoase": "qpoases", "qpoases": "qpoases", "osqp": "osqp", "hpipm": "hpipm"}
        settings = dict(SOLVER_ARGS["qp"], condensed=qp_condesed,
                        compile_data=qp_compile_data, profile=qp_profile)
        if solver != "qp":
            try:
                settings["backend"] = backends[solver.lower()]
            except KeyError:
                raise ValueError(f'{dynamics} requires solver="qpOASE", "OSQP", "HPIPM", or "qp"') from None
        return "qp", settings
    if solver not in ("fatrop", "ipopt", "osqp"):
        raise ValueError('Nonlinear dynamics require solver="fatrop", "ipopt", or lowercase "osqp" (SQP)')
    return solver, SOLVER_ARGS[solver]


def mpc_loop(ocp):
    """Time identical full-step boundaries for QP, FATROP and IPOPT.

    Timed: parameter/warm-start update, input packing, optimization, constraint
    evaluation, solution retraction and next-state calculation. Initialization,
    compilation/loading, console output and visualization are excluded.
    """
    if log_mode not in ("original", "detailed"):
        raise ValueError('log_mode must be "original" or "detailed"')
    solve_times = []
    optimization_times = []
    control_step_times = []
    integrate_state = ocp.dyn.state_integrate()
    constr_viol = []
    x_init = ocp.x_nom
    ocp.update_params(x_init, 0.)

    solver_kind, solver_settings = get_solver_configuration()
    ocp.init_solver(solver_kind, solver_settings)
    if compile_solver:
        ocp.compile_solver()
    is_nlp = solver_kind in ("fatrop", "ipopt")
    if is_nlp:
        solver_function = (ca.external("solver_function", "codegen/lib/" + load_compiled_solver)
                           if load_compiled_solver else ocp.solver_function)

    for k in range(mpc_loops):
        step_start = time.perf_counter()
        ocp.update_params(x_init, k * dt_min)
        if is_nlp:
            optimization_start = time.perf_counter()
            solver_params = ocp.get_solver_params()
            solve_start = time.perf_counter()
            sol_x = solver_function(*solver_params)
            solve_end = time.perf_counter()
            solve_time = solve_end - solve_start
            optimization_time = solve_end - optimization_start
            stacked_params = ocp.opti.value(ocp.opti.p)
            g, lbg, ubg = ocp.g_data(sol_x, stacked_params)
            cv = float(ocp.constr_viol_norm_inf(g, lbg, ubg))
            ocp.retract_stacked_sol(sol_x, retract_all=False)
        else:
            if solver_kind == "qp":
                ocp.solve(retract_all=False, verbose=False)
            else:
                # Legacy SQP emits Python diagnostics inside solve; defer them
                # until after timing, as for the other solver paths.
                import contextlib
                import io
                diagnostics = io.StringIO()
                with contextlib.redirect_stdout(diagnostics):
                    ocp.solve(retract_all=False)
            solve_time = ocp.solve_time
            optimization_time = ocp.solve_time
            cv = float(ocp.constr_viol)
        x_init = integrate_state(x_init, ocp.DX_prev[1])
        step_time = time.perf_counter() - step_start
        control_step_times.append(step_time)
        optimization_times.append(optimization_time)
        solve_times.append(solve_time)
        constr_viol.append(cv)

        # Console I/O is outside the common measurement interval in both modes.
        if log_mode == "original":
            print("Solve time (ms): ", solve_time * 1000)
            print("CV (inf norm): ", cv)
        else:
            print(f"MPC step: {step_time * 1000:.2f} ms, "
                  f"optimization: {optimization_time * 1000:.2f} ms, nonlinear CV={cv:.3g}")
            if solver_kind == "qp":
                ocp.print_qp_stats()
            elif not is_nlp:
                print(diagnostics.getvalue(), end="")

    # Legacy Solve time excludes NLP input packing; QP includes QP preparation.
    # Use mpc_step_times for comparisons with the common full-step boundary.
    ocp.solve_times = np.asarray(solve_times)
    ocp.mpc_step_times = np.asarray(control_step_times)
    ocp.optimization_times = np.asarray(optimization_times)
    ocp.mpc_constraint_violations = np.asarray(constr_viol)
    T = sum(float(ocp.opti.value(dt)) for dt in ocp.dts)
    print("************** STATS **************")
    if log_mode == "original":
        print("Avg solve time (ms): ", np.mean(ocp.solve_times) * 1000)
        print("Std solve time (ms): ", np.std(ocp.solve_times) * 1000)
    else:
        print("Avg full MPC step (ms): ", np.mean(ocp.mpc_step_times) * 1000)
        print("Std full MPC step (ms): ", np.std(ocp.mpc_step_times) * 1000)
        print("Avg optimization time (ms): ", np.mean(ocp.optimization_times) * 1000)
    print("Avg CV (inf norm): ", np.mean(ocp.mpc_constraint_violations))
    print("Horizon length (s): ", T)
    return ocp


def main():
    get_solver_configuration()  # Validate model/solver selection before building the OCP.

    # Initialize robot
    robot.set_gait_sequence(gait_type, gait_period)
    robot_instance = robot.robot
    model = robot.model
    data = robot.data
    q0 = robot.q0
    print("Robot model: ", model)

    pin.computeAllTerms(model, data, q0, np.zeros(model.nv))

    # Setup OCP
    ocp = make_ocp(
        dynamics=dynamics,
        dyn_args=DYN_ARGS[dynamics],
        robot=robot,
        nodes=nodes,
        tau_nodes=tau_nodes,
        warm_start=warm_start,
    )
    ocp.set_time_params(dt_min, dt_max)
    ocp.set_swing_params(swing_height, swing_vel_limits)
    ocp.set_tracking_targets(base_vel_des, arm_vel_des, arm_force_des)

    # Run MPC
    ocp = mpc_loop(ocp)

    if plot:
        # Plot joint positions, velocities, torques
        if hasattr(ocp, "tau_sol"):
            tau_j_sol = ocp.tau_sol
        else:
            # Compute from RNEA
            tau_j_sol = []
            for k in range(len(ocp.q_sol)):
                q = ocp.q_sol[k].flatten()
                v = ocp.v_sol[k].flatten()
                a = ocp.a_sol[k].flatten()
                forces = ocp.forces_sol[k].flatten()

                tau_rnea = ocp.dyn.rnea_dynamics()(q, v, a, forces)
                tau_rnea = np.array(tau_rnea).flatten()
                tau_j = tau_rnea[6:]
                tau_j_sol.append(tau_j)

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        labels = ["FL hip", "FL thigh", "FL calf", "FR hip", "FR thigh", "FR calf",
                  "RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf",
                  "Arm 1", "Arm 2", "Arm 3", "Arm 4"]

        axs[0].set_title("Joint positions (q)")
        for j in range(robot.nj):
            # Ignore base (quaternion)
            axs[0].plot([q[7 + j] for q in ocp.q_sol], label=labels[j])
        axs[0].set_xlabel("Time step")
        axs[0].set_ylabel("Position (rad)")

        axs[1].set_title("Joint velocities (v)")
        for j in range(robot.nj):
            # Ignore base
            axs[1].plot([v[6 + j] for v in ocp.v_sol], label=labels[j])
        axs[1].set_xlabel("Time step")
        axs[1].set_ylabel("Velocity (rad/s)")

        axs[2].set_title("Joint torques (tau)")
        for j in range(robot.nj):
            axs[2].plot([tau[j] for tau in tau_j_sol], label=labels[j])
        axs[2].set_xlabel("Time step")
        axs[2].set_ylabel("Torque (Nm)")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))

        plt.tight_layout(rect=[0, 0, 0.88, 1])  # adjust for legend
        plt.show()

    # Visualize robot
    robot_instance.initViewer()
    robot_instance.loadViewerModel("pinocchio")
    robot_instance.display(q0)
    viewer = robot_instance.viewer
    for _ in range(50):
        for (q, forces) in zip(ocp.q_sol, ocp.forces_sol):
            robot_instance.display(q)
            visualize_forces(viewer, robot, model, data, q, forces)
            time.sleep(dt_min)


if __name__ == "__main__":
    main()
