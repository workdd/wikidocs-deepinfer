DeepInfer의 분산 실행은 "attention은 data parallel, MoE는 expert parallel"이라는 한 문장으로 요약된다. 이 장은 프로세스가 어떻게 뜨고, 프로세스 그룹이 어떻게 잘리며, rank끼리 무엇을 어떤 통로로 주고받는지 따라간다. 이어서 prefill과 decode를 다른 엔진으로 나누는 PD 분리가 공개 코드에서 어디까지 구현되어 있는지 확인하고, 배포 문서가 밝힌 운영 병렬 구성을 인용한다.

## 공개 범위 확인

README는 완전한 서비스에 필요한데 이번 공개에 들어 있지 않은 것을 세 가지로 적는다([`README.md#L38-L42`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L38-L42)).

- KV 캐시 시스템
- VE/Prefill/Decode 분리 (엔진 쪽 기능은 대부분 구현됨)
- prefill, decode 양쪽의 DP 복제본 간 로드밸런싱

그래서 이 장에서 다루는 코드는 두 층으로 나뉜다. 하나의 엔진 묶음 안에서 DP, CP, EP rank가 협력하는 부분은 코드가 있다. 엔진 사이에서 KV 캐시를 넘기고 요청을 배분하는 부분은 훅(빈 메서드)만 있고 구현은 없다. 후자에 대해서는 배포 문서의 서술만 인용한다.

## 이 장에서 읽는 파일

| 파일 | 줄 수 | 역할 |
|---|---|---|
| `deepinfer/config.py` (`ParallelConfig`) | 33 (해당 클래스) | DP, CP, EP, ENP 크기와 world size 계산 |
| `deepinfer/worker/base.py` | 357 | 워커 초기화 순서, 분산 초기화, CP follower 루프 |
| `deepinfer/model_executor/parallel_utils/parallel_state.py` | 230 | 프로세스 그룹 생성 |
| `deepinfer/model_executor/parallel_utils/communication_op.py` | 82 | CP 그룹 브로드캐스트 |
| `deepinfer/model_executor/parallel_utils/cp.py` | 54 | CP all-gather 버퍼 |
| `deepinfer/distributed/shm_broadcast.py` | 461 | 공유 메모리 고리 버퍼 메시지 큐 |
| `deep_cluster/hardware/*`, `deep_cluster/topology/*` | 548, 108 | 가속기 탐지, NUMA와 CPU 바인딩 |
| `deepinfer/engine/engine_type.py`, `llm_engine.py` | 17, 1,029 | NORMAL, PROMPT, COMPLETION 엔진과 PD 훅 |

## 정리

- 엔진 하나가 DP rank 하나다. 엔진 안에서 CP 크기만큼 워커가 뜨고, cp_rank 0인 driver만 스케줄러와 샘플링을 한다. EP 그룹은 DP×CP rank의 연속 구간이다
- 기본 프로세스 그룹은 gloo이고, GPU 그룹은 `einops.rearrange`로 rank 배열을 접어 따로 만든다. lazy 통신기는 더미 all_reduce로 미리 초기화한다
- CP 그룹 안에서는 메서드 이름을 공유 메모리 고리 버퍼(vLLM 유래)로, 메타데이터 텐서를 NCCL broadcast로, prefill 중간 텐서를 DeepEP `BucketBuffer` all-gather로 보낸다. CP 그룹은 한 노드 안에 있어야 한다
- DP rank들은 C++ 스케줄러의 ALL/ANY 투표로 스텝 종류(prefill/decode)를 맞춘다. EP 그룹 안에서 스텝 모양을 같게 유지하기 위해서다
- 엔진은 NORMAL, PROMPT, COMPLETION 세 종류이고, PD 분리용 KV 전송은 `LLMEngine`의 빈 훅으로만 남아 있다. 공개본으로 PD 분리 서비스를 조립할 수는 없다
- 운영 구성(prefill EP2~8, decode EP64~128, TP와 CP 미사용, 최소 단위 64+64 GPU)은 배포 문서 인용이며 이 책에서 재현하지 않았다
