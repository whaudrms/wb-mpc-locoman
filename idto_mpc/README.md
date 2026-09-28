# B2 + Z1용 IDTO MPC

기존 `main.py`의 CasADi/Fatrop 경로와 별도로 실행되는 IDTO 백엔드입니다.
기존 Pinocchio 상태를 입력받아 관절 위치·속도 참조와 feedforward 토크를 반환합니다.
원본 URDF와 `/home/tony/4_idto` 소스는 수정하지 않습니다.

## 실행

프로젝트 루트에서 실행합니다. 현재 머신의 `idto:latest` 이미지와
`/home/tony/4_idto/build/python_bindings`를 사용합니다.
Drake 1.30 / Python 3.10 빌드에 맞춘 실행 환경이며, 기존 conda 환경은 변경하지 않습니다.
다른 설치 경로·이미지는 `IDTO_DIR`, `IDTO_IMAGE`로 지정할 수 있습니다.

```bash
# 기본 16축 모델: 정지 자세 폐루프 시뮬레이션
bash scripts/run_idto.sh --duration 2

# 전진과 팔 끝 운동을 함께 명령
bash scripts/run_idto.sh --duration 3 --vx 0.05 --arm-velocity 0.01 0 -0.01

# 브라우저 http://localhost:7000 에서 결과 재생; Enter로 종료
bash scripts/run_idto.sh --duration 2 --meshcat

# 최적화만 실행
bash scripts/run_idto.sh --mode solve

# 수치 로그 저장: /output은 지정한 호스트 디렉터리로 연결됨
IDTO_OUTPUT_DIR=/tmp/b2-idto bash scripts/run_idto.sh --duration 2 --output /output/stand.npz

# 회귀 테스트
bash scripts/run_idto.sh --test
# 호스트-컨테이너 연결 테스트 (프로젝트 루트의 호스트 Python에서)
python3 -m unittest discover -s tests -p test_idto_client.py
```

기본 설정은 14개 구간, 고정 dt=0.04초, 0.56초 예측구간, MPC 주기 0.04초,
시뮬레이션/PD 주기 0.001초, 최적화 10회 반복입니다. 접촉 전환에서 잔차가 크면
최대 2개 추가 배치를 계산합니다. `--arm-joints 0..6`으로 활성 팔 축 수를 바꿀 수 있습니다.
`--nodes`, `--dt`, `--mpc-period`, `--sim-dt`, `--iterations`,
`--initial-iterations`, `--threads`로 계산량과 주기를 조절합니다.

시뮬레이션은 계산이 끝날 때까지 시뮬레이션 시간을 멈춥니다.
따라서 성공적인 시뮬레이션이 실시간 실행을 의미하지 않습니다.
`solve_ms`는 모델 참조 갱신과 warm start를 포함한 worker의 MPC 호출 시간이며,
`optimizer_ms`는 최적화 시간입니다. JSON 통신 지연은 포함하지 않습니다.
`deadline_misses`와 `realtime_verified=false`를 함께 출력합니다.

## 기존 wb-mpc Python 환경에서 호출

호스트에 `pyidto`나 `pydrake`를 설치하지 않고 `idto_client.py`를 사용할 수 있습니다.
MPC는 컨테이너에서 실행되고, NumPy 배열을 JSON-lines로 전달합니다.
이 클라이언트는 동기 호출이므로 실시간 모터 제어 스레드에 직접 넣지 않습니다.

```python
from idto_client import IDTOClient

# 기존 reduced Pinocchio model이 있으면:
# names = tuple(robot.model.names)[2:]  # universe와 free-flyer 제외
# with IDTOClient(arm_joints=4, pin_joint_names=names) as mpc:
with IDTOClient(arm_joints=4) as mpc:
    q, v = mpc.q0, mpc.v0  # 데모 초기 자세; 실제 연결에서는 측정 상태를 사용
    q_des, v_des, tau_ff, info = mpc.solve(
        q, v, current_time=0.0,
        base_vel_des=[0.05, 0, 0, 0, 0, 0],
        arm_vel_des=[0.01, 0, -0.01],
    )
    # 다음 호출은 새로운 측정 q, v와 증가한 시각을 전달
    # q_des/v_des/tau_ff의 순서는 pin_joint_names와 동일
```

Drake와 IDTO를 이미 불러올 수 있는 동일 프로세스에서는 직접 호출할 수 있습니다.

```python
from idto_mpc import B2Z1MPC, MPCConfig

mpc = B2Z1MPC(MPCConfig(arm_joints=4))
info = mpc.solve(mpc.q0_pin, mpc.v0_pin, 0.0)
q_des, v_des, tau_ff = mpc.command(0.0)
```

`command(t)`는 마지막 궤적을 보간합니다. 유효 예측구간을 넘으면 오류를 냅니다.
토크는 generalized force 전체가 아니라 입력 관절 순서로 선택한 구동 관절 토크입니다.
시뮬레이터에서는 `tau_ff + Kp*(q_des-q) + Kd*(v_des-v)`를 계산하고
URDF effort limit으로 제한합니다. 하드웨어 드라이버나 ROS 토픽으로 명령을 전송하는 코드는 없습니다.

## 상태 정의

기본 모델은 다리 12축 + 팔 `joint1..joint4`, 총 16축입니다.
`joint5`, `joint6`, `jointGripper`는 0에서 고정합니다.
관절 순서는 기본적으로 FL, FR, RL, RR의 hip/thigh/calf, 그 다음 팔 순서입니다.
각 Drake joint index는 이름으로 조회하며, 임의 Pinocchio 순서는 `pin_joint_names`로 지정합니다.

| 항목 | Pinocchio 입출력 경계 | Drake 내부 |
|---|---|---|
| q (23) | `[xyz, qx qy qz qw, joints]` | 부유 베이스 `[qw qx qy qz, xyz]`, 관절은 이름으로 매핑 |
| v (22) | `[linear_body, angular_body, joint_vel]` | 부유 베이스 `[angular_world, linear_world]`, 관절은 이름으로 매핑 |
| 상태 x (45) | q와 v를 별도로 전달 | q와 v를 별도로 전달 |
| 관절 명령 (각 16) | `q_des, v_des, tau_ff` | actuator/generalized-force mapping으로 변환 |

Quaternion을 정규화하고 연속 MPC 호출에서 부호를 맞춥니다.
베이스 속도는 quaternion 회전행렬로 좌표계를 변환합니다.
6축 팔 설정에서는 q=25, v=24, 구동 관절=18입니다.

## 목표와 접촉의 의미

- `base_vel_des=[vx, vy, 0, 0, 0, yaw_rate]`: body frame의 평면 속도 명령입니다.
  높이는 초기 standing 참조, roll/pitch는 0으로 추종합니다.
- `arm_vel_des=[vx,vy,vz]`: body frame에서 베이스에 대한 팔 끝의 상대 속도입니다.
  `gripperCenter`의 Jacobian을 사용하는 감쇠 역기구학으로 관절 참조를 생성합니다.
  기존 OCP의 Cartesian 속도 등식제약과 달리 **soft tracking**입니다.
- 0이 아닌 `arm_force_des`는 명시적으로 오류를 냅니다. 물체 접촉과 힘 추종은 미구현입니다.
- 최적화는 IDTO의 접촉 모델로 접촉을 결정합니다. 기존 trot/walk 스케줄,
  swing 높이·시간을 강제하지 않습니다. 정해진 보행 패턴을 보장하지 않습니다.
- 접촉 기하는 발의 0.032m sphere 4개와 평지뿐입니다. 자가충돌, 몸체/팔 접촉,
  장애물 및 조작 물체는 모델에 포함하지 않습니다. 원본 질량·관성·visual은 유지합니다.
- SRDF standing 자세에서 발의 예상 정적 침투량에 맞게 초기 base z를 보정합니다.
  이는 데모 초기화이며, 측정 상태에 높이 보정을 적용하지 않습니다.
- 기존의 가변 시간격자는 고정 시간격자로 대체했습니다.

IDTO 기본 문제에는 관절 위치·속도·토크의 hard bound가 없습니다.
참조와 출력 명령은 제한하며, 계산된 전체 궤적의 제한 위반을 검사합니다.
위치 0.001rad, 속도 0.01rad/s, 토크 0.01Nm의 수치 허용오차를 넘거나,
부유 베이스 generalized-force 잔차 최대값이 `MPCConfig.max_base_residual=5.0`
(성분별 Nm/N)을 넘으면 결과를 거부합니다. 이는 최적화 문제에 hard bound를 추가한 것과 다릅니다.
기본 Python 바인딩이 solver 종료 상태를 반환하지 않아, 결과의 유한성·크기·잔차를 직접 검사합니다.

원본 `base_inertia`의 관성 주값이 triangle inequality를 위반한다는 Drake 경고가 있습니다.
측정값 근거 없이 관성을 바꾸지 않았으므로 모델 보정이 별도로 필요합니다.

## 이 머신에서의 검증 (2026-09-28)

- 좌표계 회전, 관절 순서 교환, actuator 등록, 0/4/6축 팔 모델,
  초기 상태 피드백, quaternion 부호 변화, warm start, 팔 끝 참조 관련 테스트 8개 통과.
- 호스트 Python → 컨테이너 MPC → 16축 명령 반환 성공.
  미지원 힘 명령 오류 후 다음 정상 요청도 성공.
- Meshcat 시작·짧은 시뮬레이션·기록 재생 게시·종료 경로 실행 성공.
  브라우저 렌더링을 사람이 확인한 결과는 아닙니다.
- 기본 14-node 설정의 정지 자세 2초 시험: 최저 base 높이 약 0.5364m,
  토크 포화 0회, 최대 base 잔차 약 0.0004.
- 기본 설정으로 vx=0.05m/s, 팔 상대 속도=[0.01,0,-0.01]m/s의 3초 시험:
  최저 base 높이 약 0.5307m, 최대 |roll/pitch| 약 0.0134rad(0.77도),
  토크 포화 0회. 최종 base 위치는 약 [0.0676,-0.0352,0.5332]m입니다.
  전진 명령의 이상적 이동량 0.15m보다 작고 횡방향 오차도 있어 속도 추종 튜닝이 필요합니다.
  팔은 참조 생성과 결합 시뮬레이션을 검사했으며 정확한 Cartesian 속도 추종은 검증하지 않았습니다.
- 위 3초 시험에서 평균 MPC 호출 약 307ms, 최대 약 674ms로 40ms 주기를 충족하지 못했습니다.
- `--nodes 8 --iterations 3 --threads 2` 경량 설정의 동일 명령 2초 시험:
  평균 약 44.6ms, 최대 약 121.5ms, 50회 중 34회 deadline 초과.
  최종 전진 이동은 약 0.0159m로 추종 성능도 더 낮았습니다.

현재 완료 범위는 **B2+Z1 모델과 기존 상태 형식을 연결한 실행 가능한 MPC·시뮬레이션 백엔드**입니다.
실기 배포 수준의 실시간성, 보행/팔 목표 추종 성능, 비평면 접촉, 힘 제어는 검증 범위 밖입니다.
