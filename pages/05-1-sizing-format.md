블록 개수 산정과 토큰당 바이트 수를 정하는 코드를 본다.

## 블록 개수 산정: 프로파일링

블록 개수는 워커가 모델을 한 번 돌려 본 뒤 정한다. [`worker/base.py`의 `profile_num_available_blocks`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L211-L268)가 그 일을 한다.

### 코드 분석: profile_num_available_blocks

state 캐시를 먼저 빼고 남는 메모리를 블록 크기로 나누는 부분을 본다.

```python
available_kv_cache_memory = total_gpu_memory * self.cache_config.gpu_memory_utilization - peak_allocated_memory

cache_block_size = CacheEngine.get_cache_block_size(
    self.cache_config.block_size,
    self.model_runner.model.cfg,
)
# 减去预留的固定大小的 state cache
state_size = math.prod(CacheEngine.get_state_shape(self.model_runner.model.cfg, self.model_config.max_num_seqs))
state_size *= 2  # state store & copy buffer, 所以要开两倍大小
available_kv_cache_memory -= state_size
num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
```

- `gpu_memory_utilization`의 기본값은 [`EngineArgs`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/arg_utils.py#L34-L36)에서 `0.90`이다. vLLM과 같은 이름, 같은 의미다
- state 크기를 두 배로 뺀다. 주석은 "state store & copy buffer"라고 적는다. 실제로 할당하는 `gpu_states`는 한 벌이므로, 나머지 한 벌은 저장 · 복사용 여유로 남겨 두는 셈이다(용도는 코드 주석 기준, 사용처는 미검증)
- 마지막에 모든 rank의 값을 [`all_reduce(SUM)`한 뒤 world size로 나누고 짝수로 내린다](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L264-L267). 최솟값이 아니라 평균을 쓴다는 점이 눈에 띈다

블록 하나가 몇 토큰을 담는지는 사용자가 고르지 않는다. [`ModelMetadata.resolved_tokens_per_block`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/model_metadata_utils.py#L133-L140)이 `128`을 돌려주고, `EngineArgs.tokens_per_block`을 지정하면 [이 값과 같은지 assert만 한다](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/arg_utils.py#L83-L86). 같은 파일의 `kvcache_align_to`도 `128`(바이트)이다.

## 토큰 하나의 바이트: KV 캐시 포맷

블록 크기를 계산하려면 토큰 하나가 몇 바이트인지부터 알아야 한다. 포맷 정의는 [`tilelang_ops/cache_common.py`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L109-L155)에 모여 있다.

### 코드 분석: V41_FORMAT

main · SWA KV의 기본 포맷을 본다.

```python
V41_FORMAT = KVCacheFormat(
    quantized_dtype="e4m3",
    quantized_elements=512,
    elements_per_sf=32,
    bf16_elements=0,
    do_cast=True,
    sf_format="ue8m0",
    note="V41, main/swa kv.",
    token_interleaved=True,
)
```

`bytes_per_token`은 [양자화 데이터 바이트 + scale 바이트](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L99-L101)다. 위 포맷이면 512 × 1B + (512 / 32) × 1B = 528B다. `token_interleaved=True`는 토큰마다 `[data | scale]`을 붙여 저장한다는 뜻이고, [`KVCache` docstring](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L205-L218)도 같은 숫자(FP4 288B, MXFP8 528B)를 적어 둔다.

| 포맷 | 쓰는 곳 | 원소 / scale 단위 | scale 형식 | 토큰당 바이트 |
|---|---|---|---|---|
| `V41_FORMAT` | main KV, SWA KV | e4m3 512개 / 32개당 1 | UE8M0 | 528 |
| `V41_FP4_FORMAT` | `use_fp4_sparse_kv` 일 때 main KV | e2m1 512개 / 16개당 1 | e4m3 | 288 |
| `FP4_INDEX_KV_FORMAT` | indexer K | e2m1 128개 / 32개당 1 | UE8M0 | 68 |
| `MXFP8_INDEX_KV_FORMAT` | indexer K (지원 목록) | e4m3 128개 / 32개당 1 | UE8M0 | 132 |
| `FP8_FLOAT32_SF_INDEX_KV_FORMAT` | indexer K (NVIDIA만) | e4m3 128개 / 128개당 1 | fp32 | 132 |

토큰당 바이트 칸은 위 `bytes_per_token` 공식에 각 포맷의 필드를 넣어 계산한 값이다. 어느 scale 단위를 쓸지는 [`derive_main_kv_cache_num_per_channels`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/model_metadata_utils.py#L37-L39)가 고른다. SWA는 항상 32, main KV는 `use_fp4_sparse_kv`가 켜져 있으면 16이다.

cache_engine 쪽 계산은 이 포맷을 그대로 가져다 쓴다. [`_kv_bytes_per_token`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L46-L56)은 `kv_lora_rank + qk_rope_head_dim == 512`를 assert하고 포맷의 `bytes_per_token`을 돌려준다. indexer 쪽 [`_get_qk_topk_bytes`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/cache_engine.py#L73-L80)는 `index_head_dim // 2 + index_head_dim // 32`, 곧 FP4 두 값당 1바이트에 32채널당 scale 1바이트를 더한다.

> **주의**: 저장소의 테스트 설정은 `kv_lora_rank=448`, `qk_rope_head_dim=64`를 쓴다([`tests/worker/test_state_cache_ops.py`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/tests/worker/test_state_cache_ops.py#L670-L672)). 실제 배포 모델의 설정 파일은 저장소에 없으므로, 합이 512라는 것만 코드의 assert로 확인된다.
