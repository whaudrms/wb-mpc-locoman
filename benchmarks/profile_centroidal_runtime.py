"""Headless MPC timing with optional native data evaluation and equivalence checks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import main  # Sets the configured BLAS thread count before importing NumPy.
import numpy as np
from optimization import make_ocp
from optimization.qp_codegen import compile_functions


def run(steps, compiled, verify):
    main.robot.set_gait_sequence(main.gait_type, main.gait_period)
    o=make_ocp('centroidal_vel_qp',main.DYN_ARGS['centroidal_vel_qp'],robot=main.robot,
               nodes=main.nodes,tau_nodes=main.tau_nodes,warm_start=True)
    o.set_time_params(main.dt_min,main.dt_max)
    o.set_swing_params(main.swing_height,main.swing_vel_limits)
    o.set_tracking_targets(main.base_vel_des,main.arm_vel_des,main.arm_force_des)
    o.update_params(o.x_nom,0.)
    settings=dict(main.SOLVER_ARGS['qp'],backend='hpipm',condensed=True,compile_data=False,profile=False)
    o.init_solver('qp',settings)
    interpreted=o.qp_data
    setup=time.perf_counter()
    library=None
    if compiled:
        (o.qp_data,o.g_data),library=compile_functions([o.qp_data,o.g_data])
    compile_ms=(time.perf_counter()-setup)*1000
    x=o.x_nom
    records=[]
    for k in range(steps):
        start=time.perf_counter()
        o.update_params(x,k*main.dt_min)
        update_ms=(time.perf_counter()-start)*1000
        failure=None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                o.solve(retract_all=False)
            x=o._qp_integrate(x,o.DX_prev[1])
        except RuntimeError as error:
            failure=str(error)
        total_ms=(time.perf_counter()-start)*1000
        if verify:
            for a,b in zip(interpreted(o.qp_reference,o.qp_params),o.qp_data(o.qp_reference,o.qp_params)):
                if a.sparsity()!=b.sparsity():raise AssertionError('Sparsity changed')
                np.testing.assert_allclose(a.nonzeros(),b.nonzeros(),atol=1e-8,rtol=1e-9)
        records.append(dict(step=k, failure=failure,update_ms=update_ms,total_ms=total_ms,
                            assembly_ms=o.qp_assembly_time*1000,condense_ms=o.qp_condense_time*1000,
                            setup_ms=o.qp_setup_time*1000,solver_ms=o.qp_solver_time*1000,
                            reported_ms=o.solve_time*1000,
                            affine_violation=None if failure else o.qp_constr_viol))
        if k%20==0 or failure:
            print(k,round(o.solve_time*1000,2),round(total_ms,2),failure,flush=True)
        if failure:break
    summary={}
    for name, group in [('all',lambda r:True),('initial_1_31',lambda r:1<=r['step']<=31),
                        ('late_90_96',lambda r:90<=r['step']<=96)]:
        valid=[r for r in records if not r['failure'] and group(r)]
        if valid:
            summary[name]=dict(count=len(valid))
            for key in ('assembly_ms','condense_ms','setup_ms','solver_ms','reported_ms','total_ms'):
                values=[r[key] for r in valid]
                summary[name][key]=dict(mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=max(values))
    return dict(compiled=compiled,compiled_library=library,compile_or_load_ms=compile_ms,
                blas_threads=main.blas_threads,nodes=main.nodes,summary=summary,records=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=98)
    parser.add_argument('--no-compile',action='store_true')
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/centroidal_runtime_optimized.json')
    args=parser.parse_args()
    result=run(args.steps,not args.no_compile,args.verify)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
