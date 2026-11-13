모델과 런타임에 남은 플랫폼 분기와 vLLM 플랫폼 계층과의 차이를 본다.

## 모델과 런타임의 잔여 플랫폼 분기

마지막으로 TileLang 바깥, 모델과 워커 코드에 남은 분기를 기능 단위로 모은다.

| 기능 | NVIDIA | Ascend | 근거 |
|---|---|---|---|
| mega MoE | `fp8xfp4`, `fp8xfp8`, `bf16xbf16` | `fp8xfp4`만 | [`mlp.py#L184-L188`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L184-L188) |
| gate | `deep_gemm.bf16_mega_gate` 융합 | 선형층 후 `moe_topk_gate` | [`mlp.py#L228`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L228) |
| shared expert와 통신 SM 분할 | `limit_num_sms` 적용 | 적용 안 함 | [`mlp.py#L602`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mlp.py#L602) |
| NVLink 도메인 판정 | DeepEP 물리 도메인 크기와 비교 | 항상 참 | [`parallel_state.py#L130-L134`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/parallel_utils/parallel_state.py#L130-L134) |
| EPLB 가중치 이동 | 도메인 안이면 `symm_mem` 직접 읽기 | P2P (`symm_mem` 미지원) | [`eplb/__init__.py#L199-L200`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/__init__.py#L199-L200) |
| mHC mega 커널 | 조건 맞으면 사용 | 사용 안 함 | [`hc.py#L100`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/hc.py#L100) |
| attention FP8 einsum 출력 | 조건 맞으면 사용 | 사용 안 함 | [`attn.py#L151`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L151) |
| 인덱서 KV 레이아웃 | 기본 | `use_ascend_kv_layout=False` 명시 (켜면 빠를 수 있으나 KV 레이아웃이 바뀜) | [`index_attn.py#L443-L444`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/index_attn.py#L443-L444) |
| Engram 저장소 | 호스트 메모리 + RDMA 읽기 | GPU(NPU) 메모리 강제 | [`environ.py#L154`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L154) |
| PDL | DeepGEMM, TileKernels에 켬 | 없음 | [`runtime_config.py#L64-L66`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/runtime_config.py#L64-L66) |
| 결정적 알고리즘 | batch invariant 모드에서만 | 항상 켬 (CI 수치 안정용 임시 조치) | [`runtime_config.py#L67-L69`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/runtime_config.py#L67-L69) |
| batch invariant 모드 | 지원 | assert로 금지 | [`runtime_config.py#L61-L63`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/runtime_config.py#L61-L63) |
| CUDA graph | `keep_graph=True`로 캡처 후 `instantiate()` | 캡처만 (인스턴스화된 graph만 있음) | [`cuda_graph_runner.py#L120-L131`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cuda_graph_runner.py#L120-L131) |
| graph 안 구간 타이머 | 지원 | 끔 | [`timer.py#L94-L115`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/timer.py#L94-L115) |
| NCCL 환경변수 | `NCCL_PXN_DISABLE=1`, `NCCL_CUMEM_ENABLE=1` 등 | 플랫폼 생성자에서 `HCCL_BUFFSIZE=16` 등 | [`runtime_config.py#L35-L40`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/runtime_config.py#L35-L40) |

README도 batch invariant 추론과 결정적 샘플링을 "NVIDIA only"로 적는다([`README.md#L27-L28`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L27-L28)). 배포 문서는 NPU에서 Engram CPU 오프로드를 아직 지원하지 않아 모든 Engram 데이터를 NPU 메모리에 둔다고 적는데([`inference_deployment_qa.md#L87`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/doc/inference_deployment_qa.md#L87)), `environ.py`의 기본값 계산이 이를 그대로 구현한다.

> **주의**: 표의 "Ascend" 열은 이 저장소 Python 코드의 분기만 보고 정리한 것이다. vendor의 `DeepEP-Ascend`, `DeepGEMM-Ascend` 안에서 어떤 기능이 빠졌는지는 읽지 않았다. 실제 Ascend 950 장비에서 실행해 확인한 내용이 아니다.

분기가 `PLATFORM.is_gpu`인지 `PLATFORM.is_npu`인지를 보면 기본 경로가 어느 쪽인지 드러난다. mega MoE, mega gate, PDL처럼 성능 기능은 `is_gpu`로 켜고, 제약은 `is_npu`로 건다. 코드의 기준선은 NVIDIA이고 Ascend가 그 위에서 빠진 기능을 하나씩 채워 가는 구조다. `TODO(NPU)` 주석은 기준 커밋의 Python 코드(vendor 제외)에 10곳 있다.

## vLLM의 플랫폼 계층과 비교

vLLM도 `vllm/platforms/`에 `Platform` 클래스를 두고 CUDA, ROCm, NPU 플러그인 등을 고른다(미검증. 이 책에서 vLLM 코드를 읽지 않았다). DeepInfer의 `Platform`은 속성 4개와 메서드 3개로 훨씬 작다. 대신 DeepInfer는 세 가지를 플랫폼 계층 밖에서 해결한다.

- vendor 커널 라이브러리를 플랫폼별 저장소로 나누되 패키지 이름과 API를 맞춰 import 문을 하나로 유지한다
- 직접 쓴 커널은 TileLang으로 두 벌 쓰고 `ops.py` 디스패처 하나로 고른다
- 아직 없는 함수는 `patch_imports`로 import 단계에서 바꾼다
