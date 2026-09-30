"""Freeze one trajectory's QPs, then replay IDENTICAL data to three solvers.

Run with one BLAS thread and the optional HPIPM library/Python paths configured;
see benchmarks/README.md. This does not change main.py or the controller backend.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import casadi as ca
import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import linprog

from benchmarks.qp_solver_adapters import ADAPTERS, canonical_qp, quality


def collect(path):
    import main
    from optimization import make_ocp
    main.robot.set_gait_sequence(main.gait_type, main.gait_period)
    o = make_ocp('centroidal_vel_qp', main.DYN_ARGS['centroidal_vel_qp'], robot=main.robot,
                 nodes=main.nodes, tau_nodes=main.tau_nodes, warm_start=True)
    o.set_time_params(main.dt_min, main.dt_max)
    o.set_swing_params(main.swing_height, main.swing_vel_limits)
    o.set_tracking_targets(main.base_vel_des, main.arm_vel_des, main.arm_force_des)
    o.update_params(o.x_nom, 0.)
    settings = dict(main.SOLVER_ARGS['qp'], backend='qpoases')
    o.init_solver('qp', settings)
    saved = {}
    original_condense = o.condense_qp
    def capture(qp):
        result = original_condense(qp)
        saved['basis'], saved['offset'] = result[1:]
        return result
    o.condense_qp = capture
    cases = []
    x = o.x_nom
    for k in range(98):
        o.update_params(x, k*main.dt_min)
        failure = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                o.solve(retract_all=False)
        except RuntimeError as error:
            failure = str(error)
        cases.append(dict(step=k, original=o.last_qp, qp=o.last_solver_qp,
                          basis=saved['basis'], offset=saved['offset'],
                          model_assembly_ms=(o.qp_build_time-o.qp_condense_time)*1000,
                          condense_ms=o.qp_condense_time*1000, generation_failure=failure))
        if k % 20 == 0 or failure:
            print('collected', k, 'failure', failure, flush=True)
        if failure:
            break
        x = np.asarray(o.dyn.state_integrate()(x, o.DX_prev[1])).ravel()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        pickle.dump(cases, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return cases


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def summarize(rows):
    summaries = {}
    for policy in ('replay', 'cold'):
        summaries[policy] = {}
        for name in ADAPTERS:
            summaries[policy][name] = {}
            groups = {'feasible': lambda r: r['expected_feasible'],
                      'initial': lambda r: r['expected_feasible'] and 1 <= r['step'] <= 31,
                      'late': lambda r: r['expected_feasible'] and r['step'] >= 90,
                      'infeasible': lambda r: not r['expected_feasible']}
            for group, predicate in groups.items():
                selected = [r for r in rows if r['policy']==policy and r['solver']==name and predicate(r)]
                if not selected:
                    continue
                valid = [r for r in selected if r['accepted']]
                summary = dict(attempts=len(selected), native_success=sum(r['success'] for r in selected),
                               accepted=len(valid), statuses=sorted(set(r['status'] for r in selected)))
                # Include all attempts in latency statistics; do not hide slow failures.
                for key in ('setup_ms', 'solve_ms', 'backend_ms'):
                    summary[key] = stats([r[key] for r in selected])
                if valid:
                    for key in ('original_primal', 'scaled_stationarity', 'scaled_complementarity', 'relative_objective_gap'):
                        summary[key] = max(r['quality'].get(key, 0.) for r in valid)
                summaries[policy][name][group] = summary
    return summaries


def revision(directory):
    try:
        return subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_report(report, path):
    summary, meta = report['summary'], report['metadata']
    labels = {'osqp': 'OSQP', 'qpoases': 'qpOASES', 'hpipm': 'HPIPM dense'}
    lines = [
        '# OSQP vs qpOASES vs HPIPM: centroidal condensed QP', '',
        '동일한 고정 QP를 재생한 비교이며, solver별 해로 서로 다른 다음 QP를 생성하지 않았다.', '',
        f"- CPU: {meta['cpu']}; affinity: {meta['affinity']}; BLAS/OpenMP: 1 thread",
        '- 모델: B2 + 팔 4관절, 14 nodes, trot, 938 원래 변수 → 154 condensed 변수',
        '- 공통 전처리: 0행 정리 후 636 제약 행 (원래 condensed 행 792개)',
        f"- replay: feasible step 0–96 × {meta['repeats']}회; infeasible step 97은 별도 집계",
        '- native success + 원래 QP 위반 ≤ 1e-3 + 정규화 stationarity ≤ 1e-6을 성공 기준으로 사용',
        '- solver 고유 허용오차는 서로 다르며 전체 설정은 JSON에 기록', '',
        '## Feasible 전체 구간: replay', '',
        '| Solver | 성공 | 준비 평균 ms | 풀이 평균 ms | 준비+풀이 평균 ms | P95 ms | 최대 ms |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name, label in labels.items():
        v = summary['replay'][name]['feasible']
        lines.append(f"| {label} | {v['accepted']}/{v['attempts']} | {v['setup_ms']['mean']:.2f} | {v['solve_ms']['mean']:.2f} | {v['backend_ms']['mean']:.2f} | {v['backend_ms']['p95']:.2f} | {v['backend_ms']['max']:.2f} |")
    prep = meta['common_preparation_ms']
    lines += ['', f"공통 비용은 별도이다: 모델/Jacobian 평가·행렬 조립 평균 {prep['model_assembly_ms']['mean']:.2f} ms, condensing {prep['condense_ms']['mean']:.2f} ms. 위 표는 전체 MPC 제어 주기 시간이 아니다.", '',
              '## 구간별 준비+풀이 평균', '',
              '| Solver | 초기 step 1–31 ms | 후반 step 90–96 ms |', '|---|---:|---:|']
    for name, label in labels.items():
        g=summary['replay'][name]
        lines.append(f"| {label} | {g['initial']['backend_ms']['mean']:.2f} | {g['late']['backend_ms']['mean']:.2f} |")
    lines += ['', '## 정확도: feasible replay에서의 최댓값', '',
              '| Solver | 원래 QP 위반 | 정규화 stationarity | 정규화 complementarity | 상대 목적함수 차이 |',
              '|---|---:|---:|---:|---:|']
    for name, label in labels.items():
        v=summary['replay'][name]['feasible']
        lines.append(f"| {label} | {v['original_primal']:.3e} | {v['scaled_stationarity']:.3e} | {v['scaled_complementarity']:.3e} | {v['relative_objective_gap']:.3e} |")
    lines += ['', '목적함수 차이는 동일 QP의 성공한 qpOASES 결과 중앙값 대비 `abs(J-Jref)/(1+abs(Jref))`이다. 제약 residual은 소거 전 QP로 복원해 계산했다.', '',
              '## Cold-start 비교', '',
              'step 0, 20, 40, 54, 80, 94, 95, 96을 각각 새 solver 인스턴스로 풀었다. 후반 표본 비중이 높으므로 replay 전체 평균과 직접 비교하지 않는다.', '',
              '| Solver | 성공 | 준비+풀이 평균 ms | P95 ms |', '|---|---:|---:|---:|']
    for name,label in labels.items():
        v=summary['cold'][name]['feasible']
        lines.append(f"| {label} | {v['accepted']}/{v['attempts']} | {v['backend_ms']['mean']:.2f} | {v['backend_ms']['p95']:.2f} |")
    lines += ['', '## Infeasible step 97', '',
              '세 solver 모두 성공한 해를 반환하지 않았다. OSQP·qpOASES는 infeasible 상태를 반환했고, HPIPM은 minimum-step으로 종료했다. HPIPM의 종료 상태 자체는 infeasibility 증명이 아니며, 원래 QP는 HiGHS dual-simplex로 별도 확인했다.', '',
              '## 해석과 범위', '',
              '- 이 설정에서는 HPIPM dense가 평균과 tail latency 모두 가장 작았다.',
              '- qpOASES는 초기 구간의 hot-start 효과가 크지만, 후반 및 cold-start에서는 시간이 증가했다.',
              '- OSQP는 후반에 반복 수가 증가했고, 같은 성공 기준 안에서도 잔차와 목적함수 오차가 더 컸다.',
              '- HPIPM은 dense-QP 모드이다. OCP 구조·Riccati·partial condensing의 이점까지 측정한 결과는 아니다.',
              '- OSQP fresh setup, qpOASES hot-start, HPIPM workspace 재사용/cold iterates라는 정책을 사용했다. Cold-start 결과도 함께 제시했다.',
              '- 전체 제어기는 변경하지 않았다. 별도 benchmark에서 optional HPIPM 라이브러리를 사용했다.', '',
              '## 재현', '',
              '[실행 방법](../README.md), [원시 결과 JSON](centroidal_solvers.json)', '',
              f"HPIPM `{meta['hpipm_commit']}`; BLASFEO `{meta['blasfeo_commit']}`.",
              f"CasADi {meta['casadi']}; OSQP {meta['osqp']}; NumPy {meta['numpy']}.", '',
              '공식 소스: [HPIPM](https://github.com/giaf/hpipm), [BLASFEO](https://github.com/giaf/blasfeo).', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/tmp/centroidal_solver_cases.pkl'))
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/results/centroidal_solvers.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--cpu', type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument('--deps-root', type=Path, default=Path('/tmp/wb_mpc_hpipm_bench'))
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    # Unpickling is intentionally restricted to a locally generated dataset.
    if args.dataset.exists():
        with args.dataset.open('rb') as stream:
            cases = pickle.load(stream)
    else:
        cases = collect(args.dataset)
    if args.collect_only:
        return
    names = list(ADAPTERS)
    solvers = {name: ADAPTERS[name]() for name in names}  # Native creation is timed in setup.
    options = {name: getattr(solver, 'options', {}) for name, solver in solvers.items()}
    options['osqp'] = dict(__import__('optimization.ocp_centroidal_vel_qp', fromlist=['OCPCentroidalVelQP']).OCPCentroidalVelQP.qp_default_options,
                           **__import__('args').SOLVER_ARGS['qp']['opts'])
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), platform.processor())
    metadata = dict(cpu=cpu_model, affinity=sorted(os.sched_getaffinity(0)),
                    python=sys.version, casadi=ca.__version__, numpy=np.__version__, osqp=osqp.__version__,
                    threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
                    hpipm_commit=revision(args.deps_root/'hpipm'), blasfeo_commit=revision(args.deps_root/'blasfeo'),
                    hpipm_mode='robust', options=options, repeats=args.repeats,
                    acceptance=dict(original_primal=1e-3, scaled_stationarity=1e-6),
                    policy='OSQP fresh setup; qpOASES persistent hotstart; HPIPM persistent workspace, warm_start=0',
                    dataset=str(args.dataset), sample_count=len(cases))
    # Determine infeasibility independently of the trajectory-generating solver.
    for case in cases:
        case['expected_feasible'] = True
        if case['generation_failure']:
            p = case['original']; eq = np.isfinite(p['l']) & (p['l']==p['u'])
            lo=np.isfinite(p['l']) & ~eq; hi=np.isfinite(p['u']) & ~eq
            lp=linprog(np.zeros(p['q'].size), A_ub=sparse.vstack([p['A'][hi],-p['A'][lo]]),
                       b_ub=np.r_[p['u'][hi],-p['l'][lo]], A_eq=p['A'][eq], b_eq=p['l'][eq],
                       bounds=(None,None), method='highs-ds', options={'presolve':False})
            case['lp_status'] = int(lp.status)
            if lp.status not in (0,2):
                raise RuntimeError(f"Unresolved feasibility for step {case['step']}: {lp.message}")
            case['expected_feasible'] = lp.status == 0
        case['canonical'] = canonical_qp(case['qp'])
    rows = []
    for policy in ('replay', 'cold'):
        selected = cases if policy=='replay' else [c for c in cases if c['step'] in (0,20,40,54,80,94,95,96)]
        for repeat in range(args.repeats):
            # Rotate execution order to reduce systematic thermal/order bias.
            for name in names[repeat % 3:] + names[:repeat % 3]:
                solver = ADAPTERS[name]()
                for case in selected:
                    if policy=='cold':
                        solver = ADAPTERS[name]()
                    result = solver.solve(case['canonical'])
                    q = quality(case['canonical'], result, case['original'], case['basis'], case['offset'])
                    accepted = bool(result['success'] and q['finite'] and q['original_primal'] <= 1e-3
                                    and q['scaled_stationarity'] <= 1e-6)
                    row = {key:value for key,value in result.items() if key not in ('x','dual')}
                    row.update(policy=policy, repeat=repeat, solver=name, step=case['step'],
                               expected_feasible=case['expected_feasible'], accepted=accepted, quality=q,
                               backend_ms=result['setup_ms']+result['solve_ms'])
                    rows.append(row)
                print(policy, repeat, name, 'finished', flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(metadata=metadata, rows=rows), indent=2))
    # Compare each objective with a successful qpOASES solve of the exact same QP.
    reference = {}
    for case in cases:
        costs = [r['quality']['objective'] for r in rows if r['step']==case['step'] and r['solver']=='qpoases' and r['accepted']]
        if costs:
            reference[case['step']] = float(np.median(costs))
    for row in rows:
        if row['accepted'] and row['step'] in reference:
            cost = reference[row['step']]
            row['quality']['relative_objective_gap'] = abs(row['quality']['objective']-cost)/(1.+abs(cost))
    metadata['common_preparation_ms'] = {key: stats([c[key] for c in cases if c['expected_feasible']])
                                       for key in ('model_assembly_ms','condense_ms')}
    metadata['dimensions'] = {str(c['step']): dict(original_variables=c['original']['q'].size,
                                                 condensed_variables=c['qp']['q'].size,
                                                 original_condensed_rows=c['qp']['A'].shape[0],
                                                 benchmark_rows=c['canonical']['A'].shape[0],
                                                 expected_feasible=c['expected_feasible'],
                                                 lp_status=c.get('lp_status')) for c in cases}
    report = dict(metadata=metadata, summary=summarize(rows), rows=rows)
    args.output.write_text(json.dumps(report, indent=2))
    write_report(report, args.output.with_suffix('.md'))
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
