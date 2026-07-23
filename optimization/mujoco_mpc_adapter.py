"""Non-invasive adapter from the existing WB-MPC OCP to a MuJoCo servo."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from args import DYN_ARGS, SOLVER_ARGS
from optimization import make_ocp


@dataclass(frozen=True)
class MPCStepResult:
    q_ref: np.ndarray
    v_ref: np.ndarray
    tau_ff: np.ndarray
    solve_time: float
    constraint_violation: float


class WholeBodyMPCAdapter:
    """Expose ``q1, v1, tau0`` without changing the original OCP classes."""

    def __init__(
        self,
        robot,
        *,
        dynamics: str = "whole_body_rnea",
        solver: str = "fatrop",
        nodes: int = 14,
        tau_nodes: int = 3,
        dt_min: float = 0.015,
        dt_max: float = 0.08,
        swing_height: float = 0.07,
        swing_vel_limits=(0.1, -0.2),
        base_vel_des=None,
        arm_vel_des=None,
        arm_force_des=None,
        warm_start: bool = True,
    ) -> None:
        if dynamics != "whole_body_rnea":
            raise ValueError(
                "The MuJoCo execution adapter currently requires "
                "whole_body_rnea because it needs explicit tau0."
            )
        self.robot = robot
        self.solver_name = solver
        self.ocp = make_ocp(
            dynamics=dynamics,
            dyn_args=DYN_ARGS[dynamics],
            robot=robot,
            nodes=nodes,
            tau_nodes=tau_nodes,
            warm_start=warm_start,
        )
        self.ocp.set_time_params(dt_min, dt_max)
        self.ocp.set_swing_params(swing_height, swing_vel_limits)
        self.ocp.set_tracking_targets(
            np.zeros(6) if base_vel_des is None else base_vel_des,
            np.zeros(3) if arm_vel_des is None else arm_vel_des,
            np.zeros(3) if arm_force_des is None else arm_force_des,
        )
        self.ocp.init_solver(solver, SOLVER_ARGS[solver])

    def update_targets(
        self,
        base_vel_des: np.ndarray,
        arm_vel_des: np.ndarray,
        arm_force_des: np.ndarray,
    ) -> None:
        self.ocp.set_tracking_targets(
            base_vel_des, arm_vel_des, arm_force_des
        )

    def step(
        self, q_pin: np.ndarray, v_pin: np.ndarray, t_current: float
    ) -> MPCStepResult:
        x_init = np.concatenate((
            np.asarray(q_pin, dtype=float).reshape(-1),
            np.asarray(v_pin, dtype=float).reshape(-1),
        ))
        if x_init.shape != (self.ocp.nx,):
            raise ValueError(
                f"Expected OCP state shape {(self.ocp.nx,)}, got "
                f"{x_init.shape}"
            )

        self.ocp.update_params(x_init, t_current)
        solver_params = self.ocp.get_solver_params()
        start = time.perf_counter()
        sol_x = self.ocp.solver_function(*solver_params)
        solve_time = time.perf_counter() - start

        stacked_params = self.ocp.opti.value(self.ocp.opti.p)
        g, lbg, ubg = self.ocp.g_data(sol_x, stacked_params)
        violation = float(
            self.ocp.constr_viol_norm_inf(g, lbg, ubg)
        )
        self.ocp.retract_stacked_sol(sol_x, retract_all=False)

        x_next = np.asarray(
            self.ocp.dyn.state_integrate()(
                x_init, self.ocp.DX_prev[1]
            )
        ).reshape(-1)
        tau_ff = np.asarray(
            self.ocp.U_prev[0][self.ocp.tau_idx:]
        ).reshape(-1)
        if tau_ff.shape != (self.robot.nj,):
            raise RuntimeError(
                f"Expected {self.robot.nj} feedforward torques, got "
                f"{tau_ff.shape}"
            )

        return MPCStepResult(
            q_ref=x_next[7:self.ocp.nq].copy(),
            v_ref=x_next[self.ocp.nq + 6:].copy(),
            tau_ff=tau_ff.copy(),
            solve_time=solve_time,
            constraint_violation=violation,
        )
