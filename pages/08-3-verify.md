draft 검증, 엔진 쪽 후처리, DFlash CUDA graph 를 본다.

## 검증: main forward 와 _verification

### 입력 배치

decode 스텝의 입력은 요청마다 `seqlen_q`개 토큰이다. C++ [`prepare_tensor_input_base.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/input_metadata/prepare_tensor_input_base.cpp#L108-L119)는 첫 칸에 직전 출력 토큰을, 나머지 `seqlen_q - 1`칸에 직전 스텝의 draft 앞부분을 넣는다. main 모델은 이 토큰들을 일반 decode처럼 한 번에 forward하고 모든 위치의 hidden을 sampler에 넘긴다. 5장에서 본 대로 블록 할당이 draft 수(`nextn_len`)까지 미리 잡고, SWA ring과 압축 state ring이 `DEEPINFER_MAX_BACKTRACKING_TOKENS`만큼 여유를 두는 것이 이 배치 때문이다. 거절된 위치의 KV는 다음 스텝이 같은 위치부터 다시 쓰면서 덮인다(추정. 명시적인 되돌림 코드는 찾지 못했고, ring 여유 크기와 위치 계산에서 읽은 내용이다).

샘플러 입력은 C++ [`prepare_sample.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/prepare_sample.cpp#L209-L226)가 만든다. 요청의 `j`번째 검증 행은 pool의 `read_slot × nextn + j`행을 읽고, 마지막 행은 새로 샘플링하는 행이라 0번 padding을 읽는다.

### 코드 분석: 행 메타데이터

```python
def prepare_verification_metadata_ref(ftp, seqlen_q, total_q):
    row_to_req = torch.repeat_interleave(
        torch.arange(ftp.shape[0], dtype=ftp.dtype, device=ftp.device),
        seqlen_q,
        output_size=total_q,
    )
    local_step = torch.arange(total_q, dtype=ftp.dtype, device=ftp.device) - ftp[row_to_req.long()]
    is_verify = local_step != seqlen_q[row_to_req.long()] - 1
    return row_to_req, local_step, is_verify
```

[`tilelang_ops/sampler.py` 13~25행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L13-L25)의 참조 구현에서 타입 힌트를 지웠다. 행마다 어느 요청의 몇 번째 칸인지, 검증 행인지(마지막 칸이 아닌지)를 표시한다. [`Sampler._verification`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L411-L457)은 이 표시와 pool의 draft logprob를 `process_sampling`에 넘겨, 샘플링과 draft별 수락 판정(`accept_result_verify`)을 한 커널에서 받는다. 이 판정이 표준 speculative sampling의 수락 · 거절 규칙이라는 것은 [Sampler docstring](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L74-L80)의 설명이고, 확률 계산은 9장에서 본다.

### 코드 분석: 수락된 prefix 와 마지막 행

```python
def get_accept_result_ref(raw_accept, golden, draft_token_ids, candidate_token_ids, is_verify, ftp, row_to_req):
    accept = raw_accept
    tokens = candidate_token_ids
    if golden is not None:
        use_golden = golden != -1
        accept = torch.where(use_golden, golden == draft_token_ids, accept)
        tokens = torch.where(use_golden, golden, tokens)
    blocked = ~(is_verify & accept)
    prefix = torch.cat([blocked.new_zeros(1, dtype=torch.int32), blocked.cumsum(0, dtype=torch.int32)])
    alive = prefix[:-1] == prefix[ftp[row_to_req.long()].long()]
    effective_accept = alive & is_verify & accept
    is_final = alive & blocked
    return tokens, effective_accept, is_final
```

[원본 227~248행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L227-L248)에서 타입 힌트와 docstring을 지웠다.

- `blocked`는 "수락되지 않았거나 마지막 행"이다. 누적합으로 요청 시작 이후 처음 막힌 지점을 찾고, 그 앞까지가 `alive`다
- `effective_accept`는 연속으로 수락된 draft 행, `is_final`은 요청마다 처음 막힌 행 하나다. 마지막 행에서 뽑은 토큰(거절 위치라면 그 자리에서 다시 뽑은 토큰)이 이번 스텝의 출력 토큰이자 다음 DFlash의 golden 토큰이 된다
- `golden`은 정답 토큰을 강제로 넣는 테스트용 입력이다. 이때는 draft가 정답과 같은지로 수락을 정한다

[`finalize_verification`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/sampler.py#L321-L400)이 마무리한다. `last_valid_offset = final_row - ftp`가 요청별 수락 draft 수다. 이 값이 그대로 `run_drafts`에 넘어가 다음 draft 블록의 시작 위치를 정한다. 같은 함수가 위치별 생성 · 검증 수 통계와 `batch_confidence_error`(수락 확률 추정치와 confidence의 절대 오차 합)를 계산한다. 이번 스텝에 EOS가 나온 요청은 [다음 draft의 confidence를 0으로 지운다](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/sampler.py#L252-L254).

## 엔진 쪽: process_MTPInfo 와 MTPVarlen

sampler 출력은 엔진의 [`_process_model_outputs`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L704-L723)에서 C++ `process_MTPInfo`로 들어간다. prefill 직후처럼 draft가 없으면 normal 엔진은 padding 토큰(`vocab_size - 1`)으로 채워 넘긴다.

### 코드 분석: process_MTPInfo

```cpp
mtp.draft_tokens.clear();
for (int j = 0; j < nextn; j++) {
    mtp.draft_tokens.push_back(next_n_token_ids_data[i][j]);
}
mtp.prev_accept_rates = std::move(mtp.accept_rates);
mtp.accept_rates.clear();
mtp.prev_confidence_raw = std::move(mtp.confidence_raw);
if (has_confidence) {
    float accept_rate = 1.0f;
    for (int j = 0; j < nextn; j++) {
        accept_rate *= confidence_data[i][j];
        /// @note modeling返回的confidence rate是每个token独立的， 使用前要做一个累乘
        mtp.accept_rates.push_back(accept_rate);
        mtp.confidence_raw.push_back(confidence_data[i][j]);
    }
}
```

[`scheduler.hpp` 1235~1252행](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L1235-L1252)에서 주석 일부를 지웠다. confidence head는 draft마다 독립 확률을 내므로, j번째 draft까지 모두 수락될 확률로 바꾸려고 누적곱을 한다. j번째 draft는 앞의 draft가 모두 수락돼야 의미가 있기 때문이다.

$$
a_j = \prod_{i=1}^{j} c_i
$$

[`MTPVarlen::schedule`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/mtp_varlen.hpp#L150-L267)이 이 $a_j$로 다음 스텝의 `seqlen_q`를 정한다. 스케줄러는 decode 대상을 고른 뒤 [바로 이 함수를 부른다](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L648-L659).

1. `decode_perf_curve`가 비어 있으면(`DEEPINFER_ENABLE_MTP_VARLEN` 꺼짐) 요청마다 `1 + draft 수`, 곧 모든 draft를 검증한다
2. 켜져 있으면 모든 요청의 모든 draft를 $a_j$ 내림차순으로 줄 세운다
3. 요청 수에서 시작해 draft를 하나씩 더하며 총 토큰 수 $B$를 늘리고, 그때의 기대 처리량을 계산한다
4. 처리량이 가장 큰 $B$를 고르고, 위에서부터 $B - (\text{요청 수})$개 draft의 요청에 `seqlen_q`를 1씩 더한다

$$
\text{throughput}(B) = \frac{N + \sum_{(j) \in \text{상위 } B-N} a_j}{T_{\text{fwd}}(\text{graph}(B)) + \text{slope} \cdot \text{graph}(B) + T_{\text{sampler}}}
$$

$N$은 요청 수, $\text{graph}(B)$는 $B$ 이상인 가장 작은 CUDA graph 크기다. 분자는 기대 출력 토큰 수(요청마다 최소 1개 + 수락 기대값), 분모는 그 graph 크기의 예상 스텝 시간이다([`update_best` 람다](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/mtp_varlen.hpp#L235-L249)).

| 시간 추정 재료 | 출처 | 갱신 |
|---|---|---|
| `_decode_perf_curve` | 워커 warm-up 의 [`profile_decode_perf`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L280-L281) (graph 크기별 forward 시간) | 시작 시 1회 |
| `_forward_time_slope` | 실측 forward 시간과 곡선의 차이를 graph 크기로 나눈 값의 중앙값 | 매 스텝 [`update_mtp_runtime`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L995-L1002) |
| `_sampler_time_by_num_seqs` | 요청 수별 sampler 시간, 최근 9개 중 5개 이상 모이면 중앙값 | 매 스텝 |

곡선은 문맥 길이 영향을 모르므로, slope 항으로 실행 중 보정한다는 것이 [헤더 주석](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/mtp_varlen.hpp#L44-L47)의 설명이다. 줄 세우기에 쓰는 $a_j$는 두 종류다. 총 토큰 수 $B$는 두 스텝 전 값(`prev_accept_rates`)으로, 요청별 배분은 직전 값(`accept_rates`)으로 정한다. ZOS처럼 스케줄과 후처리가 한 스텝 어긋나는 경우를 `schedule_cnt`와 `process_cnt`로 구분한다([주석](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/mtp_varlen.hpp#L160-L168)). confidence가 없으면 draft가 있는 요청은 1.0, 없는 요청은 0.0으로 채워 넣는다.

> **주의**: `seqlen_q`가 요청마다 달라지므로 main 모델 decode 입력은 가변 길이 배치다. CUDA graph는 총 토큰 수 기준 크기로 잡히고, 위의 `graph(B)` 계산이 그 크기 목록을 따른다. graph 캡처 방식 자체는 4장에서 다룬다.

## DFlash CUDA graph

draft 생성 전체(입력 브로드캐스트, `build_dflash_input`, DFlash 층, 샘플링, pool 저장)는 별도 CUDA graph로 잡힌다. [`DFlashCudaGraphRunner`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/dflash_cuda_graph_runner.py#L31-L91)는 요청 수 기준 크기 목록(`DEEPINFER_DFLASH_CUDAGRAPH_BATCH_SIZE_INTERVAL`, 기본 8 간격)마다 `run_drafts`를 캡처하고, replay 때는 입력을 고정 버퍼로 복사한 뒤 실행하고 결과에서 padding 요청을 잘라 낸다. `DEEPINFER_DFLASH_ENFORCE_EAGER`로 끌 수 있다.

[`run_drafts` docstring](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/models/mtp/dflash.py#L353-L357)은 CP 환경에서 driver가 아닌 rank도 graph 안에서 브로드캐스트를 받는다고 적는다. `last_valid_offset`과 `golden_tokens`는 driver rank의 sampler에만 있으므로 `cp_broadcast_multi_tensor`로 나눠 준다. 결과(draft 토큰)는 driver만 갖는다.

| 단계 | 함수 | CUDA graph |
|---|---|---|
| main 토큰 key 기록 | `DFlash.update_main_cache` | main 모델 decode graph 안 |
| 수락 판정 | `Sampler._verification` | graph 밖 (sampler) |
| draft 생성 + 저장 | `DFlash.run_drafts` | DFlash 전용 graph |
| 검증 길이 결정 | `MTPVarlen::schedule` | CPU (C++) |
