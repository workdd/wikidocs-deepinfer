실행 결과에 영향을 주는 환경 변수와 자주 생기는 문제를 정리한다.

## 실행에 영향을 주는 주요 환경 변수

DeepInfer는 설정 상당수를 `EngineArgs`가 아닌 환경 변수로 받는다. 모든 변수는 [deepinfer/environ.py](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L105-L218)의 `envs` 클래스에 모여 있고, 처음 읽을 때 한 번 해석해 캐시한다. 그래서 프로세스 안에서 나중에 값을 바꾸려면 `environ.rearm(이름)`을 불러야 한다. `EngineArgs.__post_init__`이 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`을 설정한 뒤 `rearm`을 부르는 것이 그 예다([deepinfer/engine/arg_utils.py#L48-L60](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/arg_utils.py#L48-L60)).

| 변수 | 기본값 | 정의 위치 | 영향 |
| --- | --- | --- | --- |
| `DEEPINFER_ACCELERATOR` | `nvidia` | [environ.py#L108](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L108) | `_so/` 폴더 선택. 빌드 스크립트 분기 |
| `DEEPINFER_USE_NEXTN_DECODE` | 5 | [environ.py#L136](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L136) | MTP draft 깊이. 0이면 추측 디코딩 끔. 8장 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | 꺼짐 | [environ.py#L138](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L138) | 신뢰도 기반 가변 길이 검증. 8장 |
| `DEEPINFER_ENFORCE_EAGER` | 꺼짐 | [environ.py#L161](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L161) | 켜면 CUDA graph를 쓰지 않음. 4장 |
| `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL` | 8 | [environ.py#L162](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L162) | 캡처할 graph 크기 간격. 4장 |
| `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` | 엔진 종류가 결정 | [environ.py#L186](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L186) | 비동기 스케줄링. NORMAL과 COMPLETION 엔진은 기본 켜짐. 3장 |
| `DEEPINFER_LOGGING_INTERVAL_STEP` | 100 | [environ.py#L189](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L189) | 통계 로그 간격(스텝). 예제는 10으로 낮춤 |
| `DEEPINFER_BATCH_INVARIANT` | 꺼짐 | [environ.py#L204](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L204) | 배치 구성과 무관한 결과. NVIDIA 전용 |
| `DEEPINFER_JIT_CACHE_DIR` | 없음 | [environ.py#L112](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L112) | tilelang, deep_gemm, deep_ep JIT 캐시 루트 |
| `DEEPINFER_HF_MODEL_DIR` | 없음 | [examples/_config.py#L42-L43](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/_config.py#L42-L43) | 기본 모델 id를 로컬 사본 경로로 바꿈. 명시한 `--model_name`은 덮지 않음 |
| `MASTER_ADDR`, `MASTER_PORT` | `localhost`, 23333 | [deepinfer/utils.py#L222-L230](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/utils.py#L222-L230) | 워커 분산 초기화 주소. 실제 포트는 `MASTER_PORT - 50 + engine_group_rank` |

`DEEPINFER_JIT_CACHE_DIR`을 주면 워커가 `TILELANG_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `EP_JIT_CACHE_DIR`을 그 아래 하위 폴더로 맞춘다([deepinfer/worker/runtime_config.py#L42-L51](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/runtime_config.py#L42-L51)). 첫 실행이 느린 원인 가운데 하나가 JIT 컴파일이므로, 같은 머신에서 반복 실행한다면 이 변수를 고정해 두는 편이 낫다(추정, 시간 측정은 하지 않았다).

## 실행 시 주요 문제

> **주의**: `LLM` docstring은 CUDA graph를 켠 상태에서 `CUDA error: an illegal memory access was encountered`가 나면 CUDA graph가 메모리를 넘긴 것일 수 있으니 `gpu_memory_utilization`을 낮춰 보라고 적는다([deepinfer/entrypoints/llm.py#L50-L51](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/entrypoints/llm.py#L50-L51)). KV cache 크기는 graph 캡처 전에 정해지므로 graph 메모리가 남은 공간을 넘을 수 있다. 계산 과정은 4장에서 본다.

- **KV cache 부족 오류**: 엔진은 블록 수로 담을 수 있는 토큰 수의 80%보다 `runtime_max_model_len`이 크면 시작을 거부한다([deepinfer/engine/llm_engine.py#L441-L451](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L441-L451)). `max_model_len`을 줄이거나 `gpu_memory_utilization`을 올린다.
- **포트 충돌**: 같은 노드에서 독립 작업을 둘 이상 돌리면 `MASTER_PORT`를 다르게 줘야 한다. 같은 작업 안의 독립 world는 `engine_group_rank`로 포트를 나눈다는 주석이 있다.
- **import 순서**: 예제 밖에서 엔진을 직접 만든다면 `CUDA_VISIBLE_DEVICES`를 정한 뒤에 `import torch`, `import deepinfer.entrypoints.llm` 순서를 지킨다. `deepinfer/__init__.py`가 torch를 import하지 않는 것도 이 때문이다.
- **프로파일러로 감싸기 금지**: `create_llm` 바깥을 `torch.profiler.profile()`로 감싸면 장치 가시성 설정이 깨진다. 초기화 프로파일이 필요하면 `profile_init=True`를 쓴다.
