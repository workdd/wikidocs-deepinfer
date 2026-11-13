Sampler.forward 의 흐름과 샘플링 설정값을 본다.

## Sampler.forward 코드 분석

### 진입부: logits 계산과 SO mask 준비의 중첩

무엇을 보는지: logits GEMM과 구조화 출력 mask 준비가 어떤 순서로 시작되는지.

[deepinfer/model_executor/layers/sampler.py L124-L143](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L124-L143)

```python
        hidden_states = hidden_states.move()
        selected_token_indices = sampler_inputs.selected_token_indices
        decode_inputs = sampler_inputs.decode
        is_prompt = main_input_metadata.is_prompt

        # Prepare prompt logprobs.
        selected_prompt_indices = sampler_inputs.selected_prompt_indices
        need_prompt_logprobs = is_prompt and selected_prompt_indices is not None
        if need_prompt_logprobs:
            prompt_hidden_states = gather(hidden_states, selected_prompt_indices)

        # Start preparing the structured-output mask while logits are computed.
        driver_inputs = sampler_inputs.driver
        if driver_inputs is not None:
            so_indices, so_bitmasks, so_status, so_mask_ready = self._get_so_mask(seq_metadata_list)

        # CUDA Graph model_output retains graph-batch padding; gather trims hidden_states.
        hidden_states = gather(hidden_states, selected_token_indices)
        logits = get_logits(hidden_states, embedding, self.vocab_size)
        del hidden_states
```

- `_get_so_mask`는 CPU에서 xgrammar로 bitmask를 채우고 별도 CUDA stream으로 복사를 건다. 그동안 기본 stream에서는 `get_logits`의 GEMM이 돈다. mask가 실제로 필요한 시점에만 `wait_event`로 맞춘다([L163-L164](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L163-L164))
- `driver_inputs`가 `None`인 rank는 follower다. context parallel(CP)에서 rank 0만 샘플링 결과를 만들고, 나머지 rank는 embedding gather 같은 집합 통신과 MTP draft forward만 같이 돌린 뒤 빈 `SamplerOutput`을 돌려준다([L145-L161](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L145-L161))
- `get_logits`는 `bf16_gemm_nt`로 `hidden @ embedding^T`를 fp32 출력 버퍼에 바로 쓴다([sampler_utils.py L20-L28](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler_utils.py#L20-L28)). 이후 단계는 모두 fp32 logits 위에서 돈다

### prefill 분기: blacklist, EOS 억제, mask

[sampler.py L166-L190](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L166-L190)은 샘플링 직전 logits에 세 가지 제약을 건다.

- blacklist 토큰을 `index_fill_`로 -inf 처리한다. 목록은 `DEEPINFER_DECODE_TOKEN_BLACKLIST`에서 받고, `'0..3,7,10..12'` 같은 구간 표기를 `parse_values`가 푼다([utils.py L29-L54](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/utils.py#L29-L54)). 평가 스크립트는 기본값으로 `0,2,129262..129264`를 넣는다(12장)
- `SamplingParams.first_n_token_no_eos=1`인 요청은 첫 출력에서 EOS를 막는다. 대상 행(`eos_suppression_rows`)은 C++ 입력 준비 단계에서 이미 계산돼 온다
- SO bitmask를 적용한다. prefill은 요청당 샘플링 행이 하나라 step 0 mask만 쓴다

decode 분기는 `_verification`으로 넘어간다. 이 함수가 MTP 검증까지 같이 처리하므로 아래 rejection sampling 절에서 본다.

## 샘플링 관련 설정값

| 이름 | 위치 | 기본값 | 의미 |
|---|---|---|---|
| `temperature` | `SamplingParams` | 1.0 | 0이면 greedy. 내부에서는 `SAMPLING_EPS=1e-5`로 바꾸고 top-p=1, top-k=-1로 고정한다 ([sampling_params.py L76-L79](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/sampling_params.py#L76-L79)) |
| `top_p` | `SamplingParams` | 1.0 | (0, 1] 범위 검사 |
| `top_k` | `SamplingParams` | -1 | -1은 끔, 0은 오류 |
| `top_logprobs` | `SamplingParams` | 0 | 0~1000 |
| `seq_seed` | `SamplingParams` | 무작위 64비트 | 해시 난수의 시드. 같으면 같은 출력 |
| `golden_token_ids` | `SamplingParams` | None | teacher forcing용 강제 토큰. -1인 자리는 샘플 결과를 쓴다 |
| `so_args` | `SamplingParams` | None | 구조화 출력 스키마 |
| `DEEPINFER_SAMPLER_FORCE_TOPK` | env | 0 | 0보다 크면 vocab 전체 대신 상위 k개 logits 공간에서만 계산 ([environ.py L157](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L157)) |
| `DEEPINFER_DECODE_TOKEN_BLACKLIST` | env | 빈 문자열 | 항상 -inf로 막을 토큰 목록 |
| `DEEPINFER_COMPUTE_SUM_SMALLER_PROBS` | env | 꺼짐 | 샘플된 토큰 기준 누적 확률 출력 (아래 참고) |
| `DEEPINFER_USE_NEXTN_DECODE` | env | 5 | MTP draft 수. 0이면 추측 디코딩 끔 ([environ.py L136](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L136)) |

> **주의**: greedy는 argmax 전용 경로가 아니라 temperature를 1e-5로 낮춘 일반 샘플링 경로를 탄다. 확률이 같은 토큰이 여럿이면 해시 난수에 따라 고르게 된다(코드 구조상 추론, 커널 동작은 미검증).
