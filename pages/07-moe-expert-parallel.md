DeepSeek-V4.1-Flash의 FFN은 전부 MoE이고, DeepInfer는 이 MoE를 expert parallel(EP)로만 나눈다. 이 장은 gate가 expert를 고르는 순간부터 DeepEP로 토큰을 보내고 DeepGEMM으로 계산해 다시 모으는 과정, 그리고 몰리는 expert를 복제해 GPU 사이 부하를 맞추는 EPLB(Expert-Parallel Load Balancer)를 코드로 따라간다.

## 이 장에서 읽는 파일

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `deepinfer/model_executor/models/v4_modules/moe.py` | 112 | HF 가중치를 담는 파라미터 컨테이너 (`FusedExperts`, `TopKGate`, `ReplicatedSharedExpert`) |
| `deepinfer/model_executor/models/v4_modules/mlp.py` | 616 | 런타임 MoE 레이어 `DeepseekMoE`. gate, dispatch, grouped GEMM, combine, mega MoE |
| `deepinfer/model_executor/eplb/__init__.py` | 434 | EPLB 런타임 상태와 재배치 주기 관리 |
| `deepinfer/model_executor/eplb/policy.py` | 140 | 복제 대상 선정과 rank 배치 탐욕 알고리즘 |
| `deepinfer/model_executor/eplb/shuffle.py` | 218 | 바뀐 배치에 맞춰 expert 가중치를 GPU 사이에서 옮기는 코드 |
| `deepinfer/model_executor/parallel_utils/userbuffer.py` | 240 | DeepEP `EPBuffer`, DeepGEMM `SymmBuffer` 같은 통신 버퍼 할당 |

프로세스 그룹(EP 그룹이 어떻게 만들어지는지)은 10장에서, NVIDIA와 Ascend에서 MoE 경로가 갈리는 이유는 11장에서 다룬다.

## expert를 세는 네 가지 단위

MoE 코드를 읽을 때 가장 먼저 헷갈리는 것이 expert 개수다. DeepInfer는 같은 expert를 네 가지 단위로 센다. [`moe.py` 모듈 docstring](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/moe.py#L1-L8)이 정의를 한 번에 적어 둔다.

| 단위 | 값 | 의미 |
|---|---|---|
| logical | `cfg.moe.n_routed_experts` | 모델이 정의한 routed expert 수. gate가 고르는 범위 |
| extra (redundant) | `cfg.moe.n_redundant_experts` | 뜨거운 expert를 복제해 둘 여분 슬롯 수 |
| physical | logical + extra | 실제로 메모리에 올라가는 expert 슬롯 수. `ep_size`로 나누어떨어져야 한다 |
| local | physical / `ep_size` | rank 하나가 가진 슬롯 수 |

shared expert는 이 셈에 들어가지 않는다. 모든 rank가 shared expert 전체를 복제해서 들고 있다.

### 코드 분석: 여분 슬롯 수 결정

무엇을 보는지: 사용자가 준 여분 개수 하한을 `ep_size` 배수가 되도록 올리는 함수. [`eplb/__init__.py#L21-L28`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/__init__.py#L21-L28)

```python
def resolve_num_extra_experts(
    num_routed_experts: int,
    min_num_extra_experts: int,
    ep_size: int,
) -> int:
    padding = -(num_routed_experts + min_num_extra_experts) % ep_size
    return min_num_extra_experts + padding
```

하한은 환경변수 `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS`(기본 0)로 들어온다. [`model_loader.finalize_model_config`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/model_loader.py#L109-L125)가 모델을 만들기 전에 이 함수를 불러 `n_redundant_experts`를 config에 써 넣고, 값이 올라갔으면 경고 로그를 남긴다([`_configure_moe_redundancy`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/model_loader.py#L83-L106)). MTP용 DFlash 모듈이 MoE 설정을 따로 가지면 그쪽도 같은 규칙으로 맞춘다.

하한 0이어도 logical 수가 `ep_size`의 배수가 아니면 여분 슬롯이 생긴다. 반대로 하한을 크게 주면 여분이 늘어 메모리를 더 쓴다. 예제 스크립트는 하한을 32로 준다([`examples/_config.py#L77-L79`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/_config.py#L77-L79)). 운영 설정은 문서가 따로 밝히는데, 이 장 끝 EPLB 설정 절에서 인용한다.

### 처음 배치

무엇을 보는지: 처음 로드할 때 physical 슬롯마다 어떤 logical expert를 놓는지 정하는 함수. [`eplb/__init__.py#L31-L39`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/eplb/__init__.py#L31-L39)

```python
def get_initial_global_physical_to_logical_map(
    num_routed_experts: int,
    num_extra_experts: int,
) -> list[int]:
    return [
        *range(num_routed_experts),
        *(expert_index % num_routed_experts for expert_index in range(num_extra_experts)),
    ]
```

앞쪽 슬롯은 logical 0번부터 차례로, 뒤쪽 여분 슬롯은 다시 0번부터 복제한다. 가중치 로더 [`_local_expert_ids`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/hugging_face/weights.py#L248-L272)도 같은 함수를 불러 `[ep_rank * local, (ep_rank + 1) * local)` 구간의 logical id만 HF 체크포인트에서 읽는다. 복제 슬롯은 같은 HF 가중치를 한 번 더 읽어 채운다. 로더와 런타임이 같은 표를 쓰므로 처음 배치에서는 가중치를 옮길 일이 없다.

## 정리

- expert는 logical, extra, physical, local 네 단위로 센다. physical은 `ep_size`의 배수가 되도록 여분 슬롯 수를 올리고, 로더와 EPLB가 같은 처음 배치 표를 쓴다
- MoE forward는 같은 NVLink 도메인 안이면 DeepGEMM mega MoE 커널 하나로, 아니면 DeepEP dispatch, DeepGEMM grouped GEMM 두 번, DeepEP combine으로 돈다. mega MoE 입력은 앞 RMSNorm이 대칭 버퍼에 바로 쓴다
- gate 커널이 EPLB의 logical→physical 표를 받아 physical 슬롯 번호를 직접 낸다. 통신 커널은 local 슬롯별 토큰 수를 받으면서 누적해 EPLB 통계를 공짜로 만든다
- EPLB 정책은 스텝별 최대 부하의 합을 줄이는 탐욕 알고리즘 두 단계(복제 대상 선정, rank 배치)다. 모든 rank가 같은 입력으로 같은 결과를 계산하므로 배치 표를 따로 주고받지 않는다
- 가중치 이동은 NVLink 도메인 안 GPU에서 `symm_mem` 미러 직접 읽기, 그 밖에서 P2P다. 배치 표는 CUDA graph 때문에 제자리 `copy_`로 갱신한다
- 재배치 주기 기본값은 0(꺼짐)이다. 운영 설정(prefill 4/100/1,000, decode 1/1,000/10,000)은 배포 문서 인용이며 이 책에서 재현하지 않았다
