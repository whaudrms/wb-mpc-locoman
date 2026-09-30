# HPIPM: OCP 유지 vs full condensing

동일한 938변수 affine QP를 사용한다. OCP 경로는 상태와 입력을 모두 유지하며,
full condensing 경로는 기존 stagewise SVD + 상태 재귀 대입으로 154변수로 줄인다.
HPIPM robust 모드, 동일 허용오차, cold primal iterates, CPU 1코어 및 BLAS 1스레드.
이 비교는 현재 Python 구현과 HPIPM의 OCP/dense 커널을 포함한다.

| 구간 | 방식 | 성공 | condensing ms | 변환/설정 ms | solve ms | 기타 ms | 합계 ms | 합계 P95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| feasible_all | ocp | 282/291 | 0.000 | 11.075 | 16.328 | 1.353 | 28.756 | 53.181 |
| feasible_all | condensed | 291/291 | 14.573 | 0.451 | 12.749 | 0.988 | 28.761 | 53.836 |
| initial_1_31 | ocp | 93/93 | 0.000 | 11.042 | 8.260 | 1.337 | 20.638 | 21.472 |
| initial_1_31 | condensed | 93/93 | 14.538 | 0.441 | 4.805 | 0.977 | 20.761 | 21.382 |
| late_90_96 | ocp | 12/21 | 0.000 | 10.985 | 74.223 | 1.387 | 86.595 | 136.305 |
| late_90_96 | condensed | 21/21 | 14.476 | 0.442 | 44.619 | 1.005 | 60.542 | 80.990 |
| both_accepted | ocp | 282/282 | 0.000 | 11.079 | 12.941 | 1.351 | 25.370 | 49.040 |
| both_accepted | condensed | 282/282 | 14.576 | 0.452 | 11.387 | 0.987 | 27.402 | 47.753 |

공통 모델 선형화/원래 QP 조립: 기존 동일 데이터 수집 시 평균 26.528 ms.
위 합계에는 공통 모델 조립을 제외하고, 형식 변환·solver 설정·solve·해 추출/복원을 포함한다.
검증용 KKT 계산과 목적함수 비교, MPC warm shift/retraction은 제외한다.
공통 모델 시간을 더한 값은 새 end-to-end 측정이 아닌 동일 모델 비용을 더한 추정치다.

OCP 실패 step: [94, 95, 96]
실패한 풀이 시간도 전체 집계에 포함했다. both_accepted는 두 경로 모두 성공한 동일 QP만 비교한다.
두 경로 성공 시 원래 좌표 목적함수 상대 차이 최대: 7.952e-11
native status 성공, 원래 제약 위반 ≤ 1e-3, scaled stationarity 및 complementarity ≤ 1e-6으로 검증했다.
OCP의 단계 내 등식은 동일 lower/upper 일반 제약으로 전달했다. 이 표현은 어려운 QP에서 수치 문제가 생길 수 있다.
현재 full condensing은 단순 상태 제거 외에 단계 내 등식 제거와 curvature scaling도 수행한다.
따라서 성공률 차이를 상태 제거 하나만의 효과로 해석해서는 안 된다.

제어기 기본 backend는 변경하지 않았다. 결과는 이 horizon, 궤적, CPU 및 구현에 한정된다.
