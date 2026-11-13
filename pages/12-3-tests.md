tests/ 디렉터리의 구성을 본다.

## tests/ 구조

pytest 설정은 `testpaths = ["tests"]`이고, `--import-mode=importlib`으로 `deepinfer` 패키지 안의 테스트 파일(커널 파일들)을 모을 때 무거운 패키지 `__init__` 연쇄를 피한다([pyproject.toml L28-L41](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/pyproject.toml#L28-L41)). 마커는 `ci`, `ci_npu`, `benchmark`, `long_context`, `gpu` 다섯 개다.

| 디렉터리 · 파일 | 대상 | 비고 |
|---|---|---|
| `tests/engine/` | FLOPs 추정기, 이미지 프롬프트 확장(`overwrite_prompt`), `Sequence` 생성 | CPU 위주 |
| `tests/ve/` | `_chunk_mm_metadata`: chunked prefill에서 이미지 경계 자르기 | CPU 텐서, `_C` 필요 |
| `tests/worker/` | state cache 저장 · 복원, userbuffer | |
| `tests/distributed/` | expert shuffle, 공유 메모리 브로드캐스트 | |
| `tests/test_cuda_sync_guards.py` | CUDA 동기화 금지 가드의 스레드 동작 | |
| `tests/models/test_offline_inference.py` | 실제 모델 decode PPL을 저장된 기준값과 정확히 비교 + SO 검사 | `ci`, `ci_npu` |
| `tests/models/test_offline_inference_long.py` | 16389 토큰 입력, 4096 토큰 출력 PPL | `long_context` |
| `tests/models/test_kl.py` | 단계별 decode logprob vs 전체 prefill logprob의 KL | VL 입력 |
| `tests/models/test_batch_invariant.py` | 배치로 돌린 결과와 단독으로 돌린 결과가 같은가 | `DEEPINFER_BATCH_INVARIANT=1` |
| `tests/models/test_ve_batch_invariant.py` | ViT 경로만의 배치 불변성 (bit 단위 일치) | |
| `tests/models/test_sampler.py` | 샘플러 분포 검정 (KS + randomized PIT) | nightly |
| `tests/models/test_eplb_policy.py` | EPLB 재배치 정책의 구조와 결정성 | CPU/CUDA |
| `deepinfer/.../tilelang_ops/*.py` | 커널별 정확도 · 벤치마크 | 커널 파일 안에 테스트가 같이 있음 |

모델 회귀 테스트는 `tests/models/_runner.py`의 `run_model_test`로 별도 프로세스 그룹을 띄우고, 제한 시간(기본 1800초)이 지나면 그룹 전체를 SIGKILL한다([_runner.py L19-L32](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/_runner.py#L19-L32)). 다중 rank 테스트가 일부만 죽어 나머지가 집합 통신에서 영원히 기다리는 상황을 정리하기 위해서다.

### PPL 정확 일치

무엇을 보는지: 오프라인 추론 회귀의 판정 기준.

[tests/models/test_offline_inference.py L153-L158](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/test_offline_inference.py#L153-L158)

```python
    for rank in LOCAL_DP_RANKS:
        expected_values = expected_ppls[rank]
        actual_values = decode_ppls[rank]
        assert len(actual_values) == len(expected_values), f"DP {rank}: golden sample count mismatch"
        for sample, (actual, expected) in enumerate(zip(actual_values, expected_values, strict=True)):
            assert actual == expected, f"DP {rank}, sample {sample}: expected PPL {expected}, got {actual}"
```

허용 오차 없이 `==`로 비교한다. 기준값은 `tests/models/offline_decode_ppl.json`에 플랫폼 · 워크로드별, DP rank별로 저장되고, 수치 변화가 의도된 것이 확인되면 `examples/refresh_offline_decode_ppl.py`로 갱신한다([examples/README.md L15](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/README.md#L15)). 해시 난수와 결정적 커널 덕분에 같은 입력 · 같은 구성이면 비트 단위로 같은 결과를 기대할 수 있다는 전제가 깔린 테스트다. `test_kl`도 같은 태도라, 기본 케이스의 `max_kl`이 0.0이다([test_kl.py L23-L35](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/test_kl.py#L23-L35), 판정 [L206](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/test_kl.py#L206)).

### 샘플러 분포 검정

PPL 일치는 "어제와 같은가"만 확인한다. 샘플러가 애초에 올바른 분포에서 뽑는지는 `test_sampler.py`가 통계적으로 확인한다.

무엇을 보는지: 한 trajectory를 균등분포 표본으로 바꾸는 randomized PIT.

[tests/models/test_sampler.py L78-L94](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/test_sampler.py#L78-L94)

```python
def pit_ks(ssp: list[float], log_probs: list[float], seed: int = 42) -> tuple[float, float, int] | None:
    """Run randomized PIT + KS test on one trajectory. Returns (ks_stat, ks_quantile, n_tokens) or None."""
    n = len(ssp)
    if n < 1 or len(log_probs) != n:
        return None
    ssp_arr = np.array(ssp, dtype=np.float64)
    p_arr = np.exp(np.array(log_probs, dtype=np.float64))
    if not (np.all(np.isfinite(ssp_arr)) and np.all(np.isfinite(p_arr)) and np.all(p_arr > 0)):
        return None
    if (ssp_arr - p_arr < -0.01).sum() > 0:
        return None
    rng = np.random.RandomState(seed)
    u = np.clip(ssp_arr - p_arr * rng.uniform(0, 1, size=n), 0, 1)
    ks_stat, ks_quantile = ks_test_uniform(u.tolist())
    if ks_stat is None:
        return None
    return ks_stat, ks_quantile, n
```

1. 엔진을 `DEEPINFER_COMPUTE_SUM_SMALLER_PROBS=1`로 띄워 토큰마다 샘플 토큰의 확률 $p$와 `sum_smaller_probs`(ssp)를 받는다(9장)
2. 이산 분포에 PIT를 적용하면 값이 균등하지 않으므로, 샘플 토큰 확률 안에서 균등 잡음을 더해 $u = \text{ssp} - p \cdot U$로 연속화한다(randomized PIT). 샘플러가 맞다면 $u \sim U(0,1)$이다
3. trajectory 하나의 $u$들에 KS 검정을 해 분위수를 얻고, trajectory 128개의 분위수가 다시 $U(0,1)$인지 KS 검정한다

샘플링 설정은 greedy, (t=1.0, p=0.95), (t=8.0, p=0.95), (t=1.0, p=1.0) 네 가지이고([L40-L46](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/test_sampler.py#L40-L46)), 최종 p값이 1e-5 이하면 실패다([L246-L247](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/models/test_sampler.py#L246-L247)). MTP 5개를 켠 상태로 돌기 때문에 rejection sampling을 거친 최종 분포가 target 분포와 맞는지까지 같이 검사한다. force_topk는 0이다.
