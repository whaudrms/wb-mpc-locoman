"""Benchmark-only adapters; they do not change the controller's backend."""
import time

import casadi as ca
import numpy as np
import osqp
from scipy import sparse

from optimization.hpipm_backend import HPIPMAdapter, canonical_qp
from args import SOLVER_ARGS
from optimization.ocp_centroidal_vel_qp import OCPCentroidalVelQP


class OSQPAdapter:
    def solve(self, qp):
        start = time.perf_counter()
        # Match production: re-equilibrate every QP, correction starts at zero.
        self.problem = osqp.OSQP()
        self.options = dict(OCPCentroidalVelQP.qp_default_options,
                            **SOLVER_ARGS['qp']['opts'])
        self.problem.setup(**{k: qp[k] for k in ('P', 'q', 'A', 'l', 'u')}, **self.options)
        self.problem.warm_start(x=np.zeros(qp['q'].size))
        setup = time.perf_counter() - start
        start = time.perf_counter()
        r = self.problem.solve()
        elapsed = time.perf_counter() - start
        return dict(x=r.x, dual=r.y, success=r.info.status_val in (1, 2),
                    status=r.info.status, iterations=int(r.info.iter),
                    setup_ms=setup*1000, solve_ms=elapsed*1000)


class QPOasesAdapter:
    def __init__(self):
        self.problem = None
        self.shape = None
        self.options = dict(SOLVER_ARGS['qp']['qpoases_opts'], error_on_fail=False)

    def solve(self, qp):
        start = time.perf_counter()
        m, n = qp['A'].shape
        if self.problem is None or self.shape != (m, n):
            self.problem = ca.conic('benchmark_qpoases', 'qpoases',
                                    {'h': ca.Sparsity.dense(n, n), 'a': ca.Sparsity.dense(m, n)},
                                    self.options)
            self.shape = (m, n)
        args = dict(h=ca.DM(qp['H'].toarray()), g=ca.DM(qp['q']),
                    a=ca.DM(qp['A'].toarray()), lba=ca.DM(qp['l']), uba=ca.DM(qp['u']))
        setup = time.perf_counter() - start
        start = time.perf_counter()
        r = self.problem(**args)
        elapsed = time.perf_counter() - start
        stats = self.problem.stats()
        return dict(x=np.asarray(r['x']).ravel(), dual=np.asarray(r['lam_a']).ravel(),
                    success=bool(stats['success']), status=stats['return_status'],
                    iterations=int(stats['iter_count']), setup_ms=setup*1000, solve_ms=elapsed*1000)




ADAPTERS = {'osqp': OSQPAdapter, 'qpoases': QPOasesAdapter, 'hpipm': HPIPMAdapter}


def quality(qp, result, original=None, basis=None, offset=None):
    # OSQP 0.6 can return object arrays of None for infeasible problems.
    x = np.asarray(result['x'], dtype=float)
    dual = np.asarray(result['dual'], dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(dual)):
        return dict(finite=False)
    Ax = qp['A'] @ x
    primal = float(max(0., np.max(qp['l']-Ax, initial=0.), np.max(Ax-qp['u'], initial=0.)))
    Hx, Aty = qp['H'] @ x, qp['A'].T @ dual
    stationarity = np.linalg.norm(Hx + qp['q'] + Aty, np.inf)
    scale = 1. + np.linalg.norm(Hx, np.inf) + np.linalg.norm(qp['q'], np.inf) + np.linalg.norm(Aty, np.inf)
    lower, upper = np.isfinite(qp['l']), np.isfinite(qp['u'])
    complementarity = max(np.max(np.abs(np.minimum(dual[lower], 0.)*(Ax[lower]-qp['l'][lower])), initial=0.),
                          np.max(np.abs(np.maximum(dual[upper], 0.)*(qp['u'][upper]-Ax[upper])), initial=0.))
    cost = float(.5*x@Hx + qp['q']@x)
    full_primal = primal
    if original is not None:
        dz = offset + basis @ x
        full_Ax = original['A'] @ dz
        full_primal = float(max(0., np.max(original['l']-full_Ax), np.max(full_Ax-original['u'])))
    return dict(finite=True, objective=cost, primal=primal, original_primal=full_primal,
                stationarity=float(stationarity), scaled_stationarity=float(stationarity/scale),
                complementarity=float(complementarity),
                scaled_complementarity=float(complementarity/(1.+abs(cost))))
