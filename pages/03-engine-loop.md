DeepInfer의 엔진 루프는 파이썬 `LLMEngine.step()`이 C++ `Scheduler`에게 이번 스텝의 배치를 받아 driver 워커에 넘기는 구조이고, 비동기 스케줄링(코드 이름 ZOS, zero-overhead scheduling)이 켜져 있으면 스텝 N의 결과 처리가 스텝 N+1의 GPU 실행과 겹친다. 이 장은 엔진이 만들어지는 순서, 요청 하나가 대기열에서 종료까지 거치는 상태, NORMAL 엔진이 prefill과 decode를 고르는 투표 규칙, 그리고 매 스텝 통계를 모으는 `StatsLogger`를 코드로 따라간다. GPU 위에서 한 스텝이 실제로 도는 과정은 4장에서 다룬다.

## 구성 요소

엔진 관련 코드는 파이썬 세 파일과 C++ 헤더 몇 개에 나뉘어 있다. 파이썬 `LLMEngine`이 C++ `Scheduler`에서 배치를 받아 driver `Worker`에 넘기고, 워커는 후처리 시점에 엔진 콜백을 불러 결과를 다시 스케줄러에 반영한다.

| 구성 요소 | 위치 | 역할 |
| --- | --- | --- |
| `LLM` | [entrypoints/llm.py#L23-L198](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/entrypoints/llm.py#L23-L198) | 오프라인 배치 진입점. 프롬프트를 토큰화해 넣고 `step()`을 반복 |
| `LLMEngine` | [engine/llm_engine.py#L130-L1029](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L130-L1029) | 워커와 스케줄러를 소유하고 스텝을 구동. PD 분리용 가상 함수를 가진 기반 클래스 |
| `Scheduler` | [csrc/engine/scheduler.hpp#L79-L1194](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L79-L1194) | waiting, prompted, running 세 대기열과 블록 할당, 선점, 종료 판정 |
| `BlockManager` | [csrc/engine/block_manager.hpp](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/block_manager.hpp) | KV 블록 풀과 시퀀스별 state 슬롯 풀. 5장 |
| `DeferredRelease` | [engine/llm_engine.py#L67-L119](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L67-L119) | 스텝 산출물(메타데이터, 출력, pin된 블록)을 한 스텝 늦춰 해제 |
| `StatsLogger` | [engine/stats_logger.py#L129-L477](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L129-L477) | 슬라이딩 창 통계, 주기적 로그, 별도 스레드 출력 |
| `ProfileManager` | [engine/profile_server.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/profile_server.py) | `DEEPINFER_PROFILE_HTTP_ENABLE=1`일 때 `POST /profile`로 스텝 프로파일 수집 |

파일 머리마다 "Derived from vLLM; modified by DeepSeek."가 붙어 있고 `LLMEngine` docstring도 vLLM의 것을 거의 그대로 쓴다. 다만 스케줄러와 시퀀스 상태는 C++로 옮겨져 있다. vLLM의 파이썬 `Scheduler`에 해당하는 역할이다(vLLM 쪽은 미검증).

## 정리

- `LLMEngine`은 CP 그룹 하나를 담당하고, driver 워커는 같은 프로세스에, 나머지 CP 워커는 데몬 프로세스로 둔다. 모든 워커 작업은 `_run_workers`가 메서드 이름을 CP 그룹에 방송하는 한 통로로 실행된다.
- 생성 순서는 워커 생성, 모델 로드, 메모리 프로파일로 블록 수 확정, KV cache 할당, CUDA graph 캡처, 스케줄러 생성이다. 블록 수는 rank 평균값이고 최대 길이가 전체 블록의 80% 안에 들어가야 시작한다.
- C++ `Scheduler`가 waiting, prompted, running 세 대기열을 관리한다. NORMAL 엔진은 한 스텝에 prefill과 decode를 섞지 않고, ALL 지점(기본 70 decode 스텝)과 ANY 지점(기본 100)에서 DP 투표로 prompt 스텝을 연다.
- decode는 출력 길이 기준으로 정렬해 블록이 모자라면 출력이 긴 시퀀스부터 선점하고, 선점된 시퀀스는 재계산 또는 종료한다. 스왑은 없다.
- 비동기 스케줄링은 inflight 토큰을 가정해 `seqlen_k`를 잡고, 결과가 오면 낙관적으로 전진한 KV 위치를 실제 수용 수로 되돌린다. 산출물 해제는 `DeferredRelease`가 한 스텝 늦춰 샘플링과 겹친다.
- `StatsLogger`는 슬라이딩 창으로 SPS, TPS, 수용률, KV 사용률, MFU를 계산하고 별도 스레드로 출력한다. MFU 분자는 스케줄러의 FLOPs 추정치다.
