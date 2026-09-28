"""Call the IDTO container from the existing wb-mpc Python environment.

No Drake, pyidto, or Pinocchio import is required in this client process.
Calls are synchronous and are intended for simulation/benchmarking, not a
hard real-time actuator thread. The worker never connects to robot hardware.
"""
import json
from pathlib import Path
import subprocess

import numpy as np


class IDTOClient:
    def __init__(self, *, arm_joints=4, nodes=14, dt=0.04, iterations=10,
                 initial_iterations=10, threads=4, pin_joint_names=None):
        script = Path(__file__).resolve().parent / "scripts/run_idto.sh"
        command = ["bash", str(script), "--server", "--arm-joints", str(arm_joints),
                   "--nodes", str(nodes), "--dt", str(dt), "--iterations", str(iterations),
                   "--initial-iterations", str(initial_iterations), "--threads", str(threads)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        try:
            ready = self._read()
            if not ready.get("ready"):
                raise RuntimeError(f"IDTO worker did not initialize: {ready}")
            worker_names = tuple(ready["joint_names"])
            self.joint_names = tuple(pin_joint_names) if pin_joint_names is not None else worker_names
            if len(set(self.joint_names)) != len(worker_names) or set(self.joint_names) != set(worker_names):
                raise ValueError("pin_joint_names must match the reduced B2+Z1 model")
            self._to_worker = [self.joint_names.index(n) for n in worker_names]
            self._from_worker = [worker_names.index(n) for n in self.joint_names]
            self.q0 = np.r_[ready["q0"][:7], np.asarray(ready["q0"])[7:][self._from_worker]]
            self.v0 = np.r_[ready["v0"][:6], np.asarray(ready["v0"])[6:][self._from_worker]]
        except Exception:
            self.close()
            raise

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("IDTO worker exited without a response; see its stderr")
        return json.loads(line)

    def solve(self, q, v, current_time, base_vel_des=None, arm_vel_des=None,
              arm_force_des=None, sample_time=None):
        n = len(self.joint_names)
        q, v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)
        if q.shape != (7 + n,) or v.shape != (6 + n,):
            raise ValueError("State dimensions do not match the configured reduced model")
        request = {"operation": "solve", "q": np.r_[q[:7], q[7:][self._to_worker]].tolist(),
                   "v": np.r_[v[:6], v[6:][self._to_worker]].tolist(), "time": float(current_time)}
        for name, value in (("base_vel_des", base_vel_des), ("arm_vel_des", arm_vel_des),
                            ("arm_force_des", arm_force_des)):
            if value is not None:
                request[name] = np.asarray(value, dtype=float).tolist()
        if sample_time is not None:
            request["sample_time"] = float(sample_time)
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        result = self._read()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "IDTO solve failed"))
        return tuple(np.asarray(result[key])[self._from_worker] for key in
                     ("q_des", "v_des", "tau_ff")) + (result["diagnostics"],)

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
        if self.process.stdout:
            self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
