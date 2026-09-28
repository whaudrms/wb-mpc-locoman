"""JSON-lines worker for callers using a different Python/Pinocchio runtime."""
import json
import sys

from idto_mpc import B2Z1MPC, MPCConfig
from main_idto import parse_args


def main():
    args = parse_args()
    mpc = B2Z1MPC(MPCConfig(arm_joints=args.arm_joints, nodes=args.nodes, dt=args.dt,
                           iterations=args.iterations, initial_iterations=args.initial_iterations,
                           threads=args.threads))
    print(json.dumps({"ready": True, "joint_names": mpc.adapter.names,
                      "q0": mpc.q0_pin.tolist(), "v0": mpc.v0_pin.tolist()}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "close":
                break
            if request.get("operation") != "solve":
                raise ValueError("Expected operation=solve")
            diagnostics = mpc.solve(request["q"], request["v"], request["time"],
                                    request.get("base_vel_des"), request.get("arm_vel_des"),
                                    request.get("arm_force_des"))
            command = mpc.command(request.get("sample_time", request["time"]))
            response = {"ok": True, "q_des": command[0].tolist(),
                        "v_des": command[1].tolist(), "tau_ff": command[2].tolist(),
                        "diagnostics": diagnostics}
        except Exception as error:
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
