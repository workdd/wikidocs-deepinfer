draft 모델 모듈 구성과 main 모델에서 넘겨받는 값을 본다.

## DFlash 모듈 구성

[`DFlash.__init__`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L82-L186)은 HF 파라미터의 main 층 뒤쪽 모듈을 모두 MTP 층으로 받는다. 전부 `DFlashNextPredictHeadHC`가 아니면 `NotImplementedError("Only DFlash speculative heads are supported")`를 낸다.

| 구성 요소 | 정체 | 하는 일 |
|---|---|---|
| `projection` + `projection_norm` | `hidden × target 층 수 → hidden` 선형 + RMSNorm | target 층 hidden 을 이어 붙인 것을 하나로 줄임 |
| `main_kv_proj` | 모든 DFlash 층의 `Wkv.A` 를 이어 붙인 선형 | main 토큰의 key 를 층 수만큼 한 GEMM 으로 계산 |
| `layers` | `DFlashDecoderLayer` × `MTP_num_layers` | draft 위치끼리, 그리고 main key 에 어텐션 + MLP(MoE 가능) |
| `post_norm` | 마지막 MTP 모듈의 `shared_head.norm` | logits 전 norm. lm head 는 main 모델 것을 공유 |
| `markov_w1`, `markov_w2` | 어휘 → `markov_rank` → 어휘 | 직전 draft 토큰에 따라 다음 위치 logits 를 보정 (`dspark_markov_rank > 0` 일 때) |
| `confidence_head` | `hidden (+ markov) → 1` 선형 + sigmoid | draft 하나가 수락될 확률 예측 |

prefill 전용 엔진(`EngineType.PROMPT`)에서는 `prefill_only=True`가 되어 main key 경로만 올리고 MLP, head, confidence head는 올리지 않는다. prefill 엔진은 draft를 만들지 않지만 decode 엔진으로 넘길 DFlash KV는 채워야 하기 때문이다(10장).

### 코드 분석: ConfidenceHead

draft마다 수락 확률을 예측하는 head를 본다.

```python
class ConfidenceHead(nn.Module):
    """Predict the acceptance probability of each draft token."""

    def __init__(self, n_embed: int, markov_rank: int, num_steps: int, weight: torch.nn.Parameter) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(input_size=n_embed + markov_rank, output_size=1, weight=weight)
        temperature = envs.DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE
        temperatures = temperature if isinstance(temperature, tuple) else (temperature,) * num_steps
        self.temperature = tuple(max(value, SAMPLING_EPS) for value in temperatures)
```

[원본 46~64행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L46-L64)에서 인자 줄바꿈을 합쳤다. forward는 TileLang 커널 [`confidence_head_fused`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/dflash.py#L10-L28)를 부르고, docstring은 계산을 `sigmoid(cat([hidden, markov_emb], -1) @ weight / temperature)`로 적는다. 입력이 norm 전 hidden이라는 점, markov 임베딩을 이어 붙인다는 점이 눈에 띈다. 이 값이 뒤에서 검증 길이 스케줄링의 재료가 된다.

## main 모델 쪽 연결: target 층 hidden 과 main key

DFlash는 main 모델 forward가 끝날 때까지 기다리지 않는다. [`DeepseekModel.__init__`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/deepseek.py#L71-L74)이 `dspark_target_layer_ids`를 층 번호 → 순번 사전으로 만들어 각 `DeepseekDecoderLayer`에 `dflash_input_index`로 넘긴다. 해당 층은 attn norm 직후 [residual 값을 `dflash_input`의 자기 칸에 복사한다](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/decoder_layer.py#L143-L145). 복사 커널 `dflash_input_store`는 hyper-connection의 `hc_mult` 축을 평균 내며 복사한다([docstring](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/dflash.py#L130-L144)).

모든 층이 끝나면 [`DeepseekModel.forward`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/deepseek.py#L220-L223)가 `dflash.update_main_cache`를 부른다.

### 코드 분석: DFlash.update_main_cache

main 토큰의 key를 DFlash 층마다 계산해 캐시에 쓰는 부분을 본다.

```python
main_hiddens = self.projection_norm(self.projection(dflash_input))
main_k = self.main_kv_proj(main_hiddens).unflatten(-1, (self.MTP_num_layers, self.qk_head_dim))
for i, layer in enumerate(self.layers):
    layer.update_main_cache(
        main_k=main_k[:, i],
        kv_cache=nextn_kv_caches[layer.layer_idx],
        main_positions=main_positions,
        slot_mapping_store_swa=input_metadata.slot_mapping_store_swa,
        mirror_cp_reverse_index=mirror_cp_reverse_index,
    )
```

[원본 219~228행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L219-L228)이다.

- main 토큰은 DFlash 층에서 query가 되지 않고 key로만 쓰인다. 그래서 MLP나 Q 투영 없이 `projection → norm → Wkv.A` 세 연산으로 끝난다
- 저장 위치는 main 모델과 같은 `slot_mapping_store_swa`다. DFlash 층의 캐시는 SWA ring 하나뿐이고, ring 크기도 main과 같다. 생성자가 [`dflash_cfg.sliding_window == window_size`를 assert](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L139-L144)하는 이유다
- DFlash 층은 5장의 캐시 엔진에서 main 층 뒤 번호(`num_main_layers + i`)로 자기 `KVCache`를 받는다

[docstring](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L204-L209)이 짚는 미묘한 점이 있다. decode 스텝에서는 검증할 draft 위치까지 모두 main key로 써 버린다. 수락 결과가 아직 없기 때문이다. 거절될 draft 자리는 곧이어 `forward_drafts`가 새 draft key로 덮어쓰므로 읽히기 전에 지워진다는 설명이다.
