"""Optional HPIPM dense/OCP backends for the single affine-QP controller."""
import ctypes
import importlib
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import linalg, sparse

DEFAULT_OPTIONS = dict(iter_max=100, tol_stat=1e-7, tol_eq=1e-8,
                       tol_ineq=1e-8, tol_comp=1e-8, reg_prim=1e-12, warm_start=0)
_NATIVE_HANDLES = []


def load_hpipm(root=None):
    """Use a configured/project-local build, or a normal system installation.

    Preloading by absolute path allows the official ctypes wrappers to resolve
    libhpipm.so without changing PYTHONPATH or LD_LIBRARY_PATH at process start.
    """
    directory = Path(root or os.environ.get('HPIPM_ROOT') or
                     Path(__file__).resolve().parents[1]/'.deps/hpipm').expanduser().resolve()
    python_dir = directory/'hpipm/interfaces/python/hpipm_python'
    try:
        if python_dir.is_dir():
            for project in ('blasfeo', 'hpipm'):
                lib = directory/'install'/project/'lib'/f'lib{project}.so'
                if not any(name == str(lib) for name, _ in _NATIVE_HANDLES):
                    _NATIVE_HANDLES.append((str(lib), ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)))
            if str(python_dir) not in sys.path:
                sys.path.insert(0, str(python_dir))
        elif root or os.environ.get('HPIPM_ROOT'):
            raise FileNotFoundError(f'HPIPM build not found at {directory}')
        module = importlib.import_module('hpipm_python')
        ctypes.CDLL('libhpipm.so')
        return module
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'HPIPM is unavailable. Build with `bash benchmarks/build_hpipm.sh .deps/hpipm` '
            'from the repository root, or set HPIPM_ROOT to an existing build. '
            f'Detail: {error}') from error


def canonical_qp(qp):
    """Remove identically zero rows satisfied to 1e-10; add a symmetric Hessian.

    A redundant 0 <= 0 row has no strict interior; it should not enter an IPM
    barrier. A 1e-10 tolerance handles subtraction roundoff in fixed-state
    bounds; original-QP residuals are still checked after solving. Nonzero rows,
    including small coefficients, are left untouched.
    """
    A = qp['A'].tocsr(copy=True)
    A.eliminate_zeros()
    keep = np.diff(A.indptr) != 0
    if np.any(qp['l'][~keep] > 1e-10) or np.any(qp['u'][~keep] < -1e-10):
        raise ValueError('Inconsistent zero constraint row')
    result = dict(qp, A=A[keep].tocsc(), l=qp['l'][keep], u=qp['u'][keep])
    result['H'] = qp['P'] + qp['P'].T - sparse.diags(qp['P'].diagonal())
    return result


class HPIPMAdapter:
    def __init__(self, mode='robust', options=None, root=None):
        load_hpipm(root)
        from hpipm_python import (hpipm_dense_qp_dim, hpipm_dense_qp, hpipm_dense_qp_sol,
                                  hpipm_dense_qp_solver_arg, hpipm_dense_qp_solver)
        self.types = (hpipm_dense_qp_dim, hpipm_dense_qp, hpipm_dense_qp_sol,
                      hpipm_dense_qp_solver_arg, hpipm_dense_qp_solver)
        self.mode = mode
        self.shape = None
        self.options = dict(DEFAULT_OPTIONS, **(options or {}))

    def solve(self, qp):
        start = time.perf_counter()
        eq = np.isfinite(qp['l']) & (qp['l'] == qp['u'])
        n = qp['q'].size
        ne, ng = int(eq.sum()), int((~eq).sum())
        shape = (n, ne, ng)
        if self.shape != shape:
            Dim, QP, Sol, Arg, Solver = self.types
            self.dim = Dim()
            for field, value in (('nv', n), ('ne', ne), ('nb', 0), ('ng', ng)):
                self.dim.set(field, value)
            self.problem = QP(self.dim)
            self.solution = Sol(self.dim)
            self.arg = Arg(self.dim, self.mode)
            for field, value in self.options.items():
                self.arg.set(field, value)
            self.solver = Solver(self.dim, self.arg)
            self.shape = shape
        self.problem.set('H', qp['H'].toarray())
        self.problem.set('g', qp['q'])
        if ne:
            self.problem.set('A', qp['A'][eq].toarray())
            self.problem.set('b', qp['l'][eq])
        lower, upper = qp['l'][~eq], qp['u'][~eq]
        self.problem.set('C', qp['A'][~eq].toarray())
        # Infinite sides are disabled, not approximated with a large bound.
        self.problem.set('lg', np.where(np.isfinite(lower), lower, 0.))
        self.problem.set('ug', np.where(np.isfinite(upper), upper, 0.))
        self.problem.set('lg_mask', np.isfinite(lower).astype(float))
        self.problem.set('ug_mask', np.isfinite(upper).astype(float))
        setup = time.perf_counter() - start
        start = time.perf_counter()
        self.solver.solve(self.problem, self.solution)
        elapsed = time.perf_counter() - start
        status = int(self.solver.get('status'))
        dual = np.zeros(qp['A'].shape[0])
        dual[~eq] = (self.solution.get('lam_ug') - self.solution.get('lam_lg')).ravel()
        if ne:
            dual[eq] = -self.solution.get('pi').ravel()
        return dict(x=self.solution.get('v').ravel(), dual=dual, success=status == 0,
                    status={0: 'success', 1: 'iteration limit', 2: 'minimum step',
                            3: 'NaN solution', 4: 'inconsistent equalities'}.get(status, str(status)),
                    iterations=int(self.solver.get('iter')), setup_ms=setup*1000, solve_ms=elapsed*1000,
                    hpipm_residuals={key: float(self.solver.get(key)) for key in
                                     ('max_res_stat', 'max_res_eq', 'max_res_ineq', 'max_res_comp')})


class HPIPMOCPAdapter:
    def __init__(self, nx, nu, dynamics_rows, mode='robust', options=None, root=None):
        load_hpipm(root)
        self.mode = mode
        from hpipm_python import (hpipm_ocp_qp_dim, hpipm_ocp_qp,
                                 hpipm_ocp_qp_sol, hpipm_ocp_qp_solver_arg,
                                 hpipm_ocp_qp_solver)
        self.types = (hpipm_ocp_qp_dim, hpipm_ocp_qp, hpipm_ocp_qp_sol,
                      hpipm_ocp_qp_solver_arg, hpipm_ocp_qp_solver)
        self.nx, self.nu = nx, nu
        self.rows = [np.asarray(r, dtype=int) for r in dynamics_rows]
        self.options = dict(DEFAULT_OPTIONS, **(options or {}))
        self.shape = None

    def solve(self, qp):
        start = time.perf_counter()
        nx, nu, N = self.nx, self.nu, len(self.rows)
        stride = nx+nu
        if qp['q'].size != N*stride+nx:
            raise ValueError('Unexpected stage variable layout')
        A = qp['A'].tocsr(copy=True)
        A.eliminate_zeros()
        H = qp['P']+qp['P'].T-sparse.diags(qp['P'].diagonal())
        local = [[] for _ in range(N+1)]
        transitions = np.concatenate(self.rows)
        if not np.all(qp['l'][transitions] == qp['u'][transitions]):
            raise ValueError('Transitions must be equalities')
        keep = np.ones(A.shape[0], dtype=bool)
        keep[transitions] = False
        for row in np.flatnonzero(keep):
            cols = A.indices[A.indptr[row]:A.indptr[row+1]]
            if not cols.size:
                if qp['l'][row] > 1e-10 or qp['u'][row] < -1e-10:
                    raise ValueError('Inconsistent zero row')
                continue
            k = int(cols[0]//stride)
            if k > N or np.any(cols//stride != k):
                raise ValueError('Nonlocal constraint')
            local[k].append(row)
        coo = H.tocoo()
        if np.any(coo.row//stride != coo.col//stride):
            raise ValueError('Nonlocal objective')
        shape = tuple(map(len, local))
        if shape != self.shape:
            Dim, QP, Sol, Arg, Solver = self.types
            self.dim = Dim(N)
            self.dim.set('nx', nx, 0, N)
            self.dim.set('nu', nu, 0, N-1)
            for k, ng in enumerate(shape):
                self.dim.set('ng', ng, k)
            self.problem, self.solution = QP(self.dim), Sol(self.dim)
            self.arg = Arg(self.dim, self.mode)
            for key, value in self.options.items():
                self.arg.set(key, value)
            self.solver = Solver(self.dim, self.arg)
            self.shape = shape
        Ds = []
        for k in range(N+1):
            p = k*stride
            width = stride if k<N else nx
            h = H[p:p+width, p:p+width].toarray()
            block = A[local[k], p:p+width].toarray()
            lower, upper = qp['l'][local[k]], qp['u'][local[k]]
            fields = dict(Q=h[:nx,:nx], q=qp['q'][p:p+nx], C=block[:,:nx],
                          lg=np.where(np.isfinite(lower), lower, 0.),
                          ug=np.where(np.isfinite(upper), upper, 0.),
                          lg_mask=np.isfinite(lower).astype(float),
                          ug_mask=np.isfinite(upper).astype(float))
            if k<N:
                if A[self.rows[k], :p].nnz or A[self.rows[k], p+stride+nx:].nnz:
                    raise ValueError('Nonlocal transition')
                transition = A[self.rows[k], p:p+stride+nx].toarray()
                D = transition[:,stride:]
                Ds.append(D)
                rec = linalg.solve(D, np.column_stack((-transition[:,:stride], qp['l'][self.rows[k]])))
                fields.update(A=rec[:,:nx], B=rec[:,nx:stride], b=rec[:,-1],
                              R=h[nx:,nx:], S=h[nx:,:nx], r=qp['q'][p+nx:p+stride], D=block[:,nx:])
            for key, value in fields.items():
                self.problem.set(key, value, k)
        setup_ms = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        self.solver.solve(self.problem, self.solution)
        solve_ms = (time.perf_counter()-start)*1000
        x = np.empty(qp['q'].size)
        dual = np.zeros(A.shape[0])
        for k in range(N+1):
            p = k*stride
            x[p:p+nx] = self.solution.get('x', k).ravel()
            dual[local[k]] = (self.solution.get('lam_ug', k)-self.solution.get('lam_lg', k)).ravel()
            if k<N:
                x[p+nx:p+stride] = self.solution.get('u', k).ravel()
                dual[self.rows[k]] = -linalg.solve(Ds[k].T, self.solution.get('pi', k).ravel())
        status = int(self.solver.get('status'))
        return dict(x=x, dual=dual, success=status==0, status=str(status),
                    iterations=int(self.solver.get('iter')), setup_ms=setup_ms, solve_ms=solve_ms,
                    hpipm_residuals={key: float(self.solver.get(key)) for key in
                                    ('max_res_stat', 'max_res_eq', 'max_res_ineq', 'max_res_comp')})
