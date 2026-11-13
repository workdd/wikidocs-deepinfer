처리량 측정 스크립트의 입력과 출력을 정리한다.

## benchmark.py 개요

`examples/benchmark.py`도 같은 `create_llm`을 쓰지만 인자가 다르다. `gpu_memory_utilization=0.95`, `max_num_seqs`는 decode 배치 크기, `max_num_batched_tokens`는 `--prefill_batch_size`로 넣는다([examples/benchmark.py#L173-L185](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/examples/benchmark.py#L173-L185)). 병렬 크기는 인자가 아닌 환경에서 정해진다. `ep_size = WORLD_SIZE × len(CUDA_VISIBLE_DEVICES)`이고 `--cp`가 이를 나눠야 한다(examples/README).

| 인자 | 기본값 | 뜻 |
| --- | --- | --- |
| `--decode_num_requests` | 64 | decode 배치 크기 |
| `--context_length` | 32768 | 요청당 컨텍스트 길이 |
| `--prefill_batch_size` | 32768 | prefill 청크 크기(`max_num_batched_tokens`) |
| `--non_cached_tokens` | 4096 | prefix cache를 쓰지 않고 새로 계산할 접미부 길이 |
| `--yoco_clip` | 0 (값 없이 주면 140) | YoCo prefill 절단 유지 토큰 수 |
| `--prompt_mode` | `text` | `text` 또는 `image` |

측정 방법, 지표 정의(prefill throughput, decode avg_SPS), sweep 스크립트는 12장에서 다룬다.
