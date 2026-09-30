"""Compare HPIPM OCP without horizon condensing against full condensing + HPIPM.

Uses the same frozen affine QPs as compare_centroidal_solvers.py. All formulation
conversion, wrapper setup, solve, and primal reconstruction time is included.
Accuracy checks are outside timings; common model assembly is reported separately.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import sparse
from benchmarks.qp_solver_adapters import HPIPMAdapter, canonical_qp, quality
from benchmarks.hpipm_ocp_adapter import HPIPMOCPAdapter
from benchmarks.compare_centroidal_solvers import stats
from optimization.qp_condensing import condense_stagewise


def structure():
    import main
    from optimization import make_ocp
    main.robot.set_gait_sequence(main.gait_type, main.gait_period)
    o = make_ocp('centroidal_vel_qp', main.DYN_ARGS['centroidal_vel_qp'], robot=main.robot,
                 nodes=main.nodes, tau_nodes=main.tau_nodes, warm_start=True)
    return o.ndx_opt, o.nu_opt[0], o._qp_dynamics_rows


def run(cases, repeats):
    nx, nu, dynamics_rows = structure()
    records = []
    for repeat in range(repeats):
        adapters = {'ocp': HPIPMOCPAdapter(nx, nu, dynamics_rows), 'condensed': HPIPMAdapter()}
        order = list(adapters) if repeat%2==0 else list(reversed(adapters))
        for case in cases:
            original = case['original']
            H = original['P']+original['P'].T-sparse.diags(original['P'].diagonal())
            for name in order:
                start = time.perf_counter()
                condense_ms = 0.
                if name == 'condensed':
                    qp, basis, offset = condense_stagewise(original, nx, nu, dynamics_rows)
                    condense_ms = (time.perf_counter()-start)*1000
                    qp = canonical_qp(qp)
                else:
                    qp = original
                r = adapters[name].solve(qp)
                x = offset+basis@r['x'] if name=='condensed' else r['x']
                total_ms = (time.perf_counter()-start)*1000
                q = quality(qp if name=='condensed' else dict(qp, H=H), r,
                            original=original if name=='condensed' else None,
                            basis=basis if name=='condensed' else None,
                            offset=offset if name=='condensed' else None)
                objective = float(.5*x@(H@x)+original['q']@x)
                accepted = bool(r['success'] and q['finite'] and q['original_primal']<=1e-3
                                and q['scaled_stationarity']<=1e-6 and q['scaled_complementarity']<=1e-6)
                records.append(dict(repeat=repeat, step=case['step'], formulation=name,
                                    expected_feasible=case['generation_failure'] is None,
                                    accepted=accepted, success=r['success'], status=r['status'],
                                    iterations=r['iterations'], condense_ms=condense_ms,
                                    setup_ms=r['setup_ms'], solve_ms=r['solve_ms'],
                                    other_ms=total_ms-condense_ms-r['setup_ms']-r['solve_ms'],
                                    total_ms=total_ms, model_assembly_ms=case['model_assembly_ms'],
                                    including_model_ms=total_ms+case['model_assembly_ms'],
                                    quality=q, original_objective=objective,
                                    hpipm_residuals=r['hpipm_residuals']))
            if case['step']%20==0 or case['step']>=95:
                print(repeat,case['step'],[(r['formulation'],r['accepted'],round(r['total_ms'],2)) for r in records[-2:]],flush=True)
    return records


def summarize(records):
    paired = {}
    for r in records:
        key = (r['repeat'], r['step'])
        paired[key] = paired.get(key, True) and r['accepted']
    summary = {}
    groups = {'feasible_all': lambda r:r['expected_feasible'],
              'initial_1_31':lambda r:1<=r['step']<=31,
              'late_90_96':lambda r:90<=r['step']<=96,
              'both_accepted':lambda r:r['expected_feasible'] and paired[(r['repeat'],r['step'])],
              'infeasible':lambda r:not r['expected_feasible']}
    for group,predicate in groups.items():
        summary[group] = {}
        for name in ('ocp','condensed'):
            selected=[r for r in records if r['formulation']==name and predicate(r)]
            if not selected:continue
            s=dict(attempts=len(selected),accepted=sum(r['accepted'] for r in selected),
                   failure_steps=sorted(set(r['step'] for r in selected if not r['accepted'])))
            for key in ('condense_ms','setup_ms','solve_ms','other_ms','total_ms','including_model_ms','iterations'):
                s[key]=stats([r[key] for r in selected])
            valid=[r for r in selected if r['accepted']]
            if valid:
                s['max_original_primal']=max(r['quality']['original_primal'] for r in valid)
                s['max_scaled_stationarity']=max(r['quality']['scaled_stationarity'] for r in valid)
            summary[group][name]=s
    return summary


def write_report(report, path):
    s=report['summary']
    lines=['# HPIPM: OCP 유지 vs full condensing', '',
           '동일한 938변수 affine QP를 사용한다. OCP 경로는 상태와 입력을 모두 유지하며,',
           'full condensing 경로는 기존 stagewise SVD + 상태 재귀 대입으로 154변수로 줄인다.',
           'HPIPM robust 모드, 동일 허용오차, cold primal iterates, CPU 1코어 및 BLAS 1스레드.',
           '이 비교는 현재 Python 구현과 HPIPM의 OCP/dense 커널을 포함한다.', '',
           '| 구간 | 방식 | 성공 | condensing ms | 변환/설정 ms | solve ms | 기타 ms | 합계 ms | 합계 P95 ms |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('feasible_all','initial_1_31','late_90_96','both_accepted'):
        for name in ('ocp','condensed'):
            v=s[group][name]
            lines.append(f"| {group} | {name} | {v['accepted']}/{v['attempts']} | "+
                         ' | '.join(f"{v[key]['mean']:.3f}" for key in ('condense_ms','setup_ms','solve_ms','other_ms','total_ms'))+
                         f" | {v['total_ms']['p95']:.3f} |")
    model=report['metadata']['model_assembly_ms']
    lines += ['', f"공통 모델 선형화/원래 QP 조립: 기존 동일 데이터 수집 시 평균 {model['mean']:.3f} ms.",
              '위 합계에는 공통 모델 조립을 제외하고, 형식 변환·solver 설정·solve·해 추출/복원을 포함한다.',
              '검증용 KKT 계산과 목적함수 비교, MPC warm shift/retraction은 제외한다.',
              '공통 모델 시간을 더한 값은 새 end-to-end 측정이 아닌 동일 모델 비용을 더한 추정치다.', '',
              f"OCP 실패 step: {s['feasible_all']['ocp']['failure_steps']}",
              '실패한 풀이 시간도 전체 집계에 포함했다. both_accepted는 두 경로 모두 성공한 동일 QP만 비교한다.',
              f"두 경로 성공 시 원래 좌표 목적함수 상대 차이 최대: {report['metadata']['max_relative_objective_gap']:.3e}",
              'native status 성공, 원래 제약 위반 ≤ 1e-3, scaled stationarity 및 complementarity ≤ 1e-6으로 검증했다.',
              'OCP의 단계 내 등식은 동일 lower/upper 일반 제약으로 전달했다. 이 표현은 어려운 QP에서 수치 문제가 생길 수 있다.',
              '현재 full condensing은 단순 상태 제거 외에 단계 내 등식 제거와 curvature scaling도 수행한다.',
              '따라서 성공률 차이를 상태 제거 하나만의 효과로 해석해서는 안 된다.', '',
              '제어기 기본 backend는 변경하지 않았다. 결과는 이 horizon, 궤적, CPU 및 구현에 한정된다.']
    path.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,default=Path('/tmp/centroidal_solver_cases.pkl'))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'benchmarks/results/hpipm_condensing.json')
    args=parser.parse_args()
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    with args.dataset.open('rb') as f:cases=pickle.load(f)
    records=run(cases,args.repeats)
    gaps=[]
    for repeat in range(args.repeats):
        for case in cases:
            pair=[r for r in records if r['repeat']==repeat and r['step']==case['step']]
            if all(r['accepted'] for r in pair):
                a,b=[r['original_objective'] for r in pair]
                gaps.append(abs(a-b)/(1+abs(b)))
    metadata=dict(repeats=args.repeats,dataset=str(args.dataset),affinity=sorted(os.sched_getaffinity(0)),
                  options=HPIPMAdapter().options, nx=28, nu=37, nodes=14,
                  original_variables=938,condensed_variables=154,
                  model_assembly_ms=stats([c['model_assembly_ms'] for c in cases if c['generation_failure'] is None]),
                  max_relative_objective_gap=max(gaps,default=None))
    report=dict(metadata=metadata,summary=summarize(records),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_report(report,args.output.with_suffix('.md'))
    print(args.output,flush=True)


if __name__=='__main__':main()
