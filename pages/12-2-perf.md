prefill 과 decode 를 나눠 재는 벤치마크와 Pareto 곡선 스크립트를 본다.

## examples/benchmark.py: prefill과 decode 분리 측정

`benchmark.py`는 엔진의 `step()`을 직접 돌리면서 시간을 잰다. 한 실행 안에서 prefill 측정과 decode 측정을 둘 다 하고, `PROFILE=1`(기본)이면 torch profiler trace도 남긴다. 설계 설명은 [examples/README.md L31-L57](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/README.md#L31-L57)에 자세히 있다.

```mermaid
flowchart LR
    A[배치 구성: 공유 prefix 1회 계산 후 fork] --> B[prefill warmup]
    B --> C[prefill 측정창]
    C --> D[prefill profile 창]
    D --> E[남은 prefill 소진]
    E --> F[decode warmup]
    F --> G[decode 로그 구간]
    G --> H[decode 측정창]
    H --> I[decode profile 창]
    I --> J[지표 출력, jsonl 추가]
```

무엇을 보는지: 측정 단계별 스텝 수와 prefill 요청 수 계산.

[examples/benchmark.py L103-L117](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/benchmark.py#L103-L117)

```python
prefill_warmup_steps = 5
prefill_measure_steps = 10
prefill_profile_steps = 2
prefill_total_steps = prefill_warmup_steps + 2 * (prefill_measure_steps + 1) + 2 * prefill_profile_steps

decode_warmup_steps = 50
decode_measure_steps = 60
decode_profile_steps = 3
decode_total_steps = decode_warmup_steps + 2 * (decode_measure_steps + 1) + 2 * (decode_profile_steps + 2)

prefill_num_requests = min(
    (prefill_total_steps * full_prefill_tokens + args.non_cached_tokens - 1) // args.non_cached_tokens,
    512,
)
batch_num_requests = max(prefill_num_requests, args.decode_num_requests)
```

- 측정 구간이 두 배인 이유: 앞 절반은 엔진 로그를 켜고(MoE 부하 통계 등을 읽기 위해) 뒤 절반은 로그를 끈 채 숫자만 잰다(README L49)
- prefill 요청 수 N은 필요한 chunk 수에서 거꾸로 계산하고 512로 자른다. decode 배치 K와는 독립이고, 배치 전체는 `max(N, K)`개 요청으로 만든다. N이 K보다 크면 남는 요청은 `max_tokens=1`로 prefill만 하고 빠진다
- 모든 요청은 소스 요청 하나가 `context_length - non_cached_tokens` 길이의 공통 prefix를 계산한 뒤 fork해서 만든다. 그래서 decode는 긴 KV를 갖고 있으면서도 prefill 측정은 non-cached 접미사만 잰다

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--decode_num_requests` | 64 | decode 배치 크기 K |
| `--context_length` | 32768 | 요청 하나의 전체 컨텍스트 |
| `--non_cached_tokens` | 4096 | prefix 캐시에 없는 접미사 길이 |
| `--prefill_batch_size` | 32768 | prefill 한 스텝 토큰 수 (`max_num_batched_tokens`) |
| `--prompt_mode` | `text` | `image`면 ViT를 포함한 VL 요청 |
| `--sampler_force_topk` | 1024 | 9장의 force_topk |
| `--enable_so` | 꺼짐 | 모든 요청에 정규식 SO 적용 |
| `--yoco_clip` | 0 (값 없이 주면 140) | YoCo prefill 끝 보존 토큰 수 |

인자 정의는 [benchmark.py L24-L65](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/benchmark.py#L24-L65). 샘플링은 temperature 0.3, top-p 0.95, `ignore_eos=True`, `top_logprobs=5`로 고정이다([L274-L283](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/benchmark.py#L274-L283)). 측정 중 EOS로 요청이 빠지지 않게 하면서 logprob 계산 비용은 포함한 구성이다.

> **주의**: benchmark의 force_topk 기본값(1024)은 평가(0)와 다르다. 처리량 숫자와 평가 점수를 같은 구성의 결과로 묶어 읽으면 안 된다.

### 지표 정의

무엇을 보는지: decode 처리량을 어떤 가정으로 환산하는지.

[examples/benchmark.py L397-L409](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/benchmark.py#L397-L409)

```python
    def measure_decode() -> dict[str, float]:
        """稳态窗口的 decode 原始量；窗口内无 flush-clean 区间时返回空 dict。"""
        clean = [ts - prev[0] for prev, (ts, flushed) in pairwise(stable_decode) if not flushed]
        if not clean:
            return {}
        avg_sps = len(clean) / sum(clean)
        return {
            "decode_window_steps": len(clean),
            "decode_avg_sps": avg_sps,  # VARLEN=0 下每 step 恒提交 1+nextn 个 token，故与接受率无关
            "decode_max_step_ms": max(clean) * 1000,
            # 满接受口径：每 step 每请求按 1 + nextn 个 token 计，换 τ 从 decode_avg_sps 重算
            "decode_tokens_per_s_per_gpu": avg_sps * args.decode_num_requests * (envs.DEEPINFER_USE_NEXTN_DECODE + 1) / cp_size,
        }
```

| 지표 | 식 | 해석 |
|---|---|---|
| prefill throughput | 측정창 토큰 합 / 측정창 시간 합 / `cp` | tokens/s/GPU |
| decode avg_SPS | 측정 구간 수 / 구간 시간 합 | 초당 decode 스텝 수 |
| decode throughput | avg_SPS × K × (1 + nextn) / `cp` | 모든 draft가 수락됐다고 가정한 tokens/s/GPU |
| max_step | 가장 느린 구간 (ms) | 정상이면 `1000 / avg_SPS` 근처 |

decode throughput은 수락률과 무관하게 스텝마다 요청당 `1 + nextn`개 토큰을 낸다고 센 상한값이다. 실제 처리량을 보려면 수락률을 반영해 다시 계산해야 하며, 스윕 그림이 그 일을 한다(아래).

측정창은 EPLB의 부하 통계 flush(EP all_gather)와 겹친 구간을 버린다. flush 시점을 `lb_flush_countdown()`으로 미리 알 수 있어, profile 창 앞에서는 스텝을 더 돌려 flush 위상을 피한다([L442-L455](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/benchmark.py#L442-L455)). README는 profiler 관련 주의를 두 가지 적는다. profiler를 끈 뒤에도 bsz=1 기준 약 7~11% 느려진 상태가 남아 그 뒤에 측정을 두면 안 되고, trace 안의 스텝 시간은 profiler 부하로 bsz=1에서 최대 50% 가까이 부풀어 SPS 계산에 쓰면 안 된다는 것이다([examples/README.md L47](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/README.md#L47)).

## examples/sweep: Pareto 곡선

`run_sweep.sh`는 한 축을 바꿔 가며 `benchmark.py`를 여러 번 실행하고(항상 `PROFILE=0`), 결과를 한 `results.jsonl`에 모은 뒤 `plot_pareto.py`로 그린다.

| 모드 | 스윕 축 | 값 | 고정 인자 |
|---|---|---|---|
| `decode` | `--decode_num_requests` | 8, 16, 32, 64, 128, 256 | `context_length`, nextn, draft 수락률, `--non_cached_tokens 4096` |
| `prefill` | `--prefill_batch_size` | 4096, 8192, 16384, 32768, 65536 | `context_length`, `--yoco_clip` (140) |
| `encoder` | 이미지 수 | 1, 2, 4, 8, 16, 32 (`ve_sweep.py` 기본) | 이미지 800×800 |

값은 [run_sweep.sh L28-L57](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/sweep/run_sweep.sh#L28-L57)과 [ve_sweep.py L27-L33](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/sweep/ve_sweep.py#L27-L33)에 있다. 스윕 점마다 `MASTER_PORT`를 하나씩 올린다. 앞 점의 TCPStore 포트가 TIME_WAIT에 남아 rendezvous가 걸리는 것을 피하기 위해서다.

무엇을 보는지: decode 곡선에 수락률을 반영하는 방법.

[examples/sweep/plot_pareto.py L58-L68](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/sweep/plot_pareto.py#L58-L68)

```python
def decode_tau(points: list[dict], draft_acceptance: float) -> float:
    """τ = 1 + nextn × a：每 step 期望接受 nextn × a 个 draft token，另加 1 个主模型 token。"""
    assert not points[0]["mtp_varlen"], "VARLEN=1 时 avg_SPS 已含接受率，不能再乘 τ；这组数据不能用本口径出图"
    return 1 + points[0]["nextn"] * draft_acceptance


def decode_axes(points: list[dict], tau: float) -> tuple[list[float], list[float], list[str]]:
    """两轴同乘 τ：一个 step 发出的 token 数既进每用户速度，也进每卡吞吐。"""
    xs = [p["decode_avg_sps"] * tau for p in points]
    ys = [p["decode_avg_sps"] * p["decode_num_requests"] * tau / p["cp"] for p in points]
    return xs, ys, [f"{p['decode_num_requests']:,}" for p in points]
```

draft 토큰별 수락률을 $a$라 하면 스텝당 기대 출력 토큰 수는 $\tau = 1 + \text{nextn} \cdot a$로 둔다. 가로축은 사용자 한 명이 보는 속도(avg_SPS × τ), 세로축은 GPU당 처리량(avg_SPS × K × τ / cp)이다. 이 환산은 draft 수락이 위치와 무관하게 독립이라고 보는 단순화이고, 연속 접두사만 받아들이는 실제 규칙(9장)과는 다르다. 수락률은 사용자가 명령행에서 주는 가정값이다(README 예시는 0.75).

> **주의**: variable-length MTP(`DEEPINFER_ENABLE_MTP_VARLEN=1`)로 잰 데이터에는 이 환산을 쓰지 못하게 assert로 막는다. 그 경우 스텝마다 검증하는 draft 수가 달라 avg_SPS에 이미 수락률이 섞이기 때문이다. 평가(`run_eval.py`)는 VARLEN을 켜고, 벤치마크 스윕은 끈 상태를 전제로 한다.

`ve_sweep.py`는 LLM 없이 VE 쪽 가중치만 올려 한 프로세스 안에서 pad → ViT → downsample → aligner만 잰다. CPU 전처리는 엔진에서 워커 프로세스가 스텝 경로 밖에서 하고, H2D는 일반 입력 전송에 묻혀 가므로 둘 다 뺀다. 처리량은 이미지당 LLM 토큰 수로 환산해 prefill 곡선과 같은 단위로 맞춘다([examples/README.md L120](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/README.md#L120)).

저장소 README의 성능 그림(prefill · decode Pareto)은 이 스윕으로 만든 것이며, 측정 환경은 prefill SM100 GPU 4장, decode 64장이고 각각 하나의 scale-up 도메인 안이라고 적혀 있다([README.md L77-L83](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L77-L83)). 그림의 수치는 이미지라 이 책에서 옮기지 않는다.
