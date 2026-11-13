스텝별 처리량과 지연을 기록하는 코드를 본다.

## StatsLogger

### 수집 항목

`StatsLogger`는 매 스텝 `StepStats` 하나를 받는다([engine/stats_logger.py#L32-L49](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L32-L49)). `log_stats=False`(`disable_log_stats=True`)면 `_process_model_outputs`가 빈 람다를 돌려주어 수집 자체를 건너뛴다.

| 창 (`Window`) | 저장하는 값 | 만드는 지표 |
| --- | --- | --- |
| `_prompt_tokens` | prompt 스텝 토큰 수 | `prompt_TPS` |
| `_generation_tokens` | 생성 토큰 수, 시퀀스 수, 배치 토큰 수 | `completion_TPS`, `cur_SPS`, `avg_SPS`, nextn 수용률 |
| `_verified_tokens` | 검증 토큰 수 | `avg_verified_len` |
| `_confidence_errors` | 배치 신뢰도 오차 | `avg_confidence_error` |
| `_moe_lb` | (평균, 최대) 전문가 토큰 수 | `moe_lb` |
| `_tflops` | 스케줄러 추정 TFLOPs | `MFU` |

창은 `(timestamp, payload, step)` 튜플 목록이고, 로그 시점에 `DEEPINFER_LOGGING_INTERVAL_STEP` 스텝보다 오래된 항목을 잘라낸다. `_moe_lb`만 최소 `DEEPINFER_MOE_LB_STAT_INTERVAL`(기본 128) 스텝을 유지한다([engine/stats_logger.py#L452-L458](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L452-L458)).

### 코드 분석: 처리량과 MFU 계산

[engine/stats_logger.py#L195-L209](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L195-L209)에서 두 지표의 분모가 다르다는 점을 본다.

```python
    def _throughput(window: Window, now: float, key: Callable[[object], float]) -> tuple[float, int]:
        """(tokens/s, total_tokens) over a window, ignoring the first sample."""
        if len(window) <= 1:
            return 0.0, 0
        total = sum(key(p) for _, p, _ in window.entries[1:])
        # (원문 주석 생략: 분모를 now 까지 늘린다는 설명)
        return total / (now - window.entries[0][0]), total

    def _mfu(self, now: float) -> float:
        entries = self._tflops.entries
        if len(entries) <= 1:
            return 0.0
        # 口径：首条仅作区间起点（值不计入），分母为首尾样本间隔、不延伸到 now（对比 _throughput；_tflops 每步都追加，实际末样本时间戳即 now）
        tflops_per_s = (sum(v for _, v, _ in entries) - entries[0][1]) / (entries[-1][0] - entries[0][0])
        return tflops_per_s / self._parallel_config.model_parallel_size / self._tflops_limit
```

두 계산 모두 창의 첫 표본은 시작 시각으로만 쓰고 값은 더하지 않는다. 처리량은 분모를 현재 시각까지 늘리므로 스텝이 멈추면 값이 시간에 따라 줄어든다. MFU는 첫 표본과 마지막 표본 사이 간격을 쓴다. 정리하면 다음과 같다.

$$
\text{MFU} = \frac{\sum_{i=2}^{n} \text{TFLOPs}_i \,/\, (t_n - t_1)}{\text{CP 크기} \times \text{TFLOPS 한도}}
$$

TFLOPS 한도는 엔진이 `get_bf16_tflops_limit() * 2`로 넘긴 값이다. 주석은 FP8 기준 최대치를 BF16의 2배로 잡았다고 설명한다([engine/llm_engine.py#L298-L305](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L298-L305)). 분자는 실측이 아니라 스케줄러가 시퀀스 길이로 추정한 FLOPs다(`FlopsEstimator`).

### 출력 시점과 위치

로그 여부는 `_should_log`가 정한다([engine/stats_logger.py#L381-L390](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L381-L390)). 첫 스텝과 그 뒤 `DEEPINFER_LOGGING_INTERVAL_STEP` 스텝마다 출력하고, PROMPT 엔진은 간격을 5분의 1로 줄인다. 0 이하로 두면 출력하지 않는다. 벤치마크가 측정 구간에서 로그를 끌 때 이 값을 0으로 둔다(examples/README).

출력은 주 스레드가 아니라 데몬 스레드 `stats_log_worker`가 한다([engine/stats_logger.py#L116-L126](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L116-L126)). 주 스레드는 문자열 목록을 `SimpleQueue`에 넣기만 하고, 로거 핸들러 락을 잡지 않는다. 출력 스레드는 rank 번호에 따라 0~35ms를 기다렸다가 쓴다. 여러 rank의 로그가 한 터미널에 섞이는 것을 줄이려는 것으로 보인다(추정).

한 번의 로그는 `[stats]` 한 줄과 `[cpu]`, `[cuda]` 타이머 줄로 구성된다([engine/stats_logger.py#L272-L340](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L272-L340)). 엔진 루프를 진단할 때 보는 필드는 다음과 같다.

| 필드 | 뜻 | 확인할 때 |
| --- | --- | --- |
| `is_prompt` | 이번 스텝이 prompt 스텝인가 | prefill 투표가 열리는 빈도 |
| `cur_SPS`, `avg_SPS` | 초당 스텝 수 | decode 속도. 벤치마크의 기본 지표 |
| `prompt_TPS`, `completion_TPS` | 초당 prompt, 생성 토큰 | 처리량 |
| `avg_nextn_accept_rate`, `nextn_accept_rate [...]` | MTP draft 위치별 수용률 | 8장 |
| `running`, `pending`, `chunked` | running 수, waiting prompt 수, 청크 진행 중 수 | 대기열 적체 |
| `num_batched_tokens [x%]` | 스텝 토큰 수와 상한 대비 비율 | 배치 채움 정도 |
| `KV_cache_usage`, `state_cache_usage` | 블록, state 슬롯 사용률 | 0.9를 넘으면 prefill 투표에서 빠진다 |
| `prefill/decode x%/y%` | NORMAL 엔진에서 prompt 스텝과 decode 스텝이 차지한 경과 시간 비율(직전 스텝 종료부터 잰 벽시계 시간, [stats_logger.py#L355-L379](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/stats_logger.py#L355-L379)) | 스텝 종류 배분 |
| `TTFT (max, mean)` | 첫 토큰 지연(ms) | 스케줄러가 도착 시각 기준으로 계산 |
| `preempted`, `aborted` | 누적 선점, 취소 수 | KV 부족 신호 |
| `MFU` | 위 식 | 스케줄러 FLOPs 추정 기준 |

`[cpu]` 줄의 항목 이름은 코드의 `cpu_timer.record("...")` 이름과 같다. `schedule:cpp`, `run_workers`, `bubble`, `prepare_input`, `launch_forward`, `pre_sampler_sync` 등을 보면 스텝 안에서 CPU 시간이 어디에 쓰이는지 알 수 있다.
