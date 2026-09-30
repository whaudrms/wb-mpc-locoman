# OSQP vs qpOASES vs HPIPM: centroidal condensed QP

> Historical result: measured before centroidal QP was aligned to the original
> NLP's delta-coordinate Euler integration, previous-node warm start and absent
> terminal joint bounds. Regenerate data to benchmark the current formulation.

동일한 고정 QP를 재생한 비교이며, solver별 해로 서로 다른 다음 QP를 생성하지 않았다.

- CPU: 12th Gen Intel(R) Core(TM) i7-12700H; affinity: [0]; BLAS/OpenMP: 1 thread
- 모델: B2 + 팔 4관절, 14 nodes, trot, 938 원래 변수 → 154 condensed 변수
- 공통 전처리: 0행 정리 후 636 제약 행 (원래 condensed 행 792개)
- replay: feasible step 0–96 × 3회; infeasible step 97은 별도 집계
- native success + 원래 QP 위반 ≤ 1e-3 + 정규화 stationarity ≤ 1e-6을 성공 기준으로 사용
- solver 고유 허용오차는 서로 다르며 전체 설정은 JSON에 기록

## Feasible 전체 구간: replay

| Solver | 성공 | 준비 평균 ms | 풀이 평균 ms | 준비+풀이 평균 ms | P95 ms | 최대 ms |
|---|---:|---:|---:|---:|---:|---:|
| OSQP | 291/291 | 23.27 | 80.27 | 103.53 | 234.25 | 2201.24 |
| qpOASES | 291/291 | 6.91 | 31.87 | 38.78 | 200.06 | 453.59 |
| HPIPM dense | 291/291 | 0.42 | 13.14 | 13.56 | 39.68 | 71.51 |

공통 비용은 별도이다: 모델/Jacobian 평가·행렬 조립 평균 26.53 ms, condensing 16.73 ms. 위 표는 전체 MPC 제어 주기 시간이 아니다.

## 구간별 준비+풀이 평균

| Solver | 초기 step 1–31 ms | 후반 step 90–96 ms |
|---|---:|---:|
| OSQP | 69.13 | 495.74 |
| qpOASES | 11.75 | 251.40 |
| HPIPM dense | 5.29 | 47.23 |

## 정확도: feasible replay에서의 최댓값

| Solver | 원래 QP 위반 | 정규화 stationarity | 정규화 complementarity | 상대 목적함수 차이 |
|---|---:|---:|---:|---:|
| OSQP | 1.845e-05 | 4.998e-08 | 4.531e-06 | 2.449e-05 |
| qpOASES | 5.777e-13 | 1.893e-14 | 2.514e-12 | 5.773e-11 |
| HPIPM dense | 2.897e-12 | 1.649e-11 | 5.747e-10 | 4.383e-08 |

목적함수 차이는 동일 QP의 성공한 qpOASES 결과 중앙값 대비 `abs(J-Jref)/(1+abs(Jref))`이다. 제약 residual은 소거 전 QP로 복원해 계산했다.

## Cold-start 비교

step 0, 20, 40, 54, 80, 94, 95, 96을 각각 새 solver 인스턴스로 풀었다. 후반 표본 비중이 높으므로 replay 전체 평균과 직접 비교하지 않는다.

| Solver | 성공 | 준비+풀이 평균 ms | P95 ms |
|---|---:|---:|---:|
| OSQP | 24/24 | 384.07 | 2176.70 |
| qpOASES | 24/24 | 365.52 | 554.00 |
| HPIPM dense | 24/24 | 29.14 | 73.02 |

## Infeasible step 97

세 solver 모두 성공한 해를 반환하지 않았다. OSQP·qpOASES는 infeasible 상태를 반환했고, HPIPM은 minimum-step으로 종료했다. HPIPM의 종료 상태 자체는 infeasibility 증명이 아니며, 원래 QP는 HiGHS dual-simplex로 별도 확인했다.

## 해석과 범위

- 이 설정에서는 HPIPM dense가 평균과 tail latency 모두 가장 작았다.
- qpOASES는 초기 구간의 hot-start 효과가 크지만, 후반 및 cold-start에서는 시간이 증가했다.
- OSQP는 후반에 반복 수가 증가했고, 같은 성공 기준 안에서도 잔차와 목적함수 오차가 더 컸다.
- HPIPM은 dense-QP 모드이다. OCP 구조·Riccati·partial condensing의 이점까지 측정한 결과는 아니다.
- OSQP fresh setup, qpOASES hot-start, HPIPM workspace 재사용/cold iterates라는 정책을 사용했다. Cold-start 결과도 함께 제시했다.
- 전체 제어기는 변경하지 않았다. 별도 benchmark에서 optional HPIPM 라이브러리를 사용했다.

## 재현

[실행 방법](../README.md), [원시 결과 JSON](centroidal_solvers.json)

HPIPM `bad1123a96136fce3cdd8bbcbe24b853d012b840`; BLASFEO `9628393214623b29e5dd2a8784f7423edbc93e04`.
CasADi 3.6.7; OSQP 0.6.7.post3; NumPy 2.5.1.

공식 소스: [HPIPM](https://github.com/giaf/hpipm), [BLASFEO](https://github.com/giaf/blasfeo).
