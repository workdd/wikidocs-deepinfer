prefill 에서 모은 KV 위에 sparse 어텐션을 계산하는 경로를 본다.

## prefill 어텐션: 모은 KV 위의 sparse fwd

prefill은 캐시에서 KV를 꺼내 BF16 버퍼 하나에 모은 뒤 sparse 어텐션을 돌린다. 버퍼 구성은 [forward 811~855행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L811-L855)에 있다.

```
kv 버퍼 = [ 캐시에 있던 SWA KV | 이번 청크 SWA KV | 캐시의 압축 KV 전체 ]
           total_k_swa_cached   total_k_swa_cur     total_k_compressed
```

- 캐시에 있던 SWA KV와 압축 KV는 `load_paged_kv_cache`가 블록 구간 목록(`load_segments_*`)을 따라 읽어 BF16으로 풀어 놓는다
- query마다 볼 위치는 SWA 쪽 `kv_swa_full_index`와 압축 쪽 `kv_ca_index`를 [`kv_index_cat`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L603-L610)으로 이어 붙인 한 줄이다. 압축 쪽 인덱스는 버퍼 안 오프셋을 더해 옮긴다
- share-kv에서 consumer 층은 source 층이 모아 둔 버퍼를 [그대로 재사용](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L723-L750)한다

### 코드 분석: prefill_attn_varlen_sparse

FlashMLA를 부르는 지점을 본다.

```python
if fused_norm_rope_cast_kwargs is not None:
    output_fp8, output_sf, _max_logits, lse = fused_norm_rope_attn_rope_cast.prefill(
        q=query,
        kv=kv_a_with_mqa,
        indices=index_topk.unsqueeze(1),
        sm_scale=softmax_scale,
        attn_sink=attn_sink,
        topk_length=index_length,
        **fused_norm_rope_cast_kwargs,
    )
    return (output_fp8, output_sf), lse

num_heads = query.shape[1]
query, attn_sink, padded_heads = _pad_attention_heads(query, 1, attn_sink)
output, _max_logits, lse = flash_mla_sparse_fwd(
    query, kv_a_with_mqa, index_topk.unsqueeze(1),
    sm_scale=softmax_scale, attn_sink=attn_sink, topk_length=index_length,
)
```

[`layers/attention_backend.py` 50~71행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/attention_backend.py#L50-L71)이다. `flash_mla`는 `vendor/FlashMLA` 서브모듈에서 오는 패키지라 이 저장소 기준 커밋에는 본체가 없고 호출부만 있다.

- 두 경로 모두 "query마다 볼 KV 위치 목록"을 받는 sparse 커널이다. dense 층도 이 커널을 쓰고, 위치 목록을 문맥 전체로 채우고 `topk_length`로 유효 길이를 알려 준다
- fused 경로(`fused_norm_rope_attn_rope_cast`)는 q norm, q RoPE, 어텐션, 출력 역 RoPE, FP8 양자화를 한 커널로 묶는다. [사용 조건](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/attn.py#L87-L92)은 NVIDIA GPU이고 head 수가 64 또는 128인 경우다. 압축 KV가 있는 dense 층은 여기에 더해 prefill 전용 엔진(`EngineType.PROMPT`)이면서 `DEEPINFER_BATCH_INVARIANT`가 꺼져 있을 때만 fused 경로를 쓴다
- FlashMLA가 받는 head 수가 64, 128뿐이라 그 외 head 수는 [`_pad_attention_heads`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/attention_backend.py#L5-L23)가 0으로 채워 맞추고 결과에서 잘라 낸다
- `attn_sink`는 head별 fp32 logit이다. [`runtime_attn_sink`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/v4_modules/mla_attn.py#L295-L299)가 HF 파라미터를 복사 없이 넘긴다. softmax 분모에 더해지는 가상의 토큰 역할로 보인다(추정, 커널 미확인)
