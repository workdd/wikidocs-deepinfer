워커가 GPU 를 잡고 KV 블록 수를 정하는 과정을 본다.

## 워커 초기화

생성자는 `_init_torch`, `init_model`, `init_dist` 순서로 돈다([worker/base.py#L102-L123](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L102-L123)).

| 단계 | 하는 일 | 근거 |
| --- | --- | --- |
| `gc.disable()` | 자동 GC를 끄고 `DEEPINFER_GC_INTERVAL_STEP`(기본 1000, COMPLETION 엔진은 10배) 스텝마다 수동 수집 | [base.py#L102-L106](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L102-L106) |
| `_init_torch` | reduced precision reduction 끄기, `expandable_segments:True`, torch 스레드 8개 | [base.py#L174-L180](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L174-L180) |
| `init_model` | NCCL 환경 변수와 JIT 캐시 경로 설정, NUMA 고정, `cuda:{local_rank}` 선택, 커널 라이브러리 설정, CPU 고정, rank별 시드 | [base.py#L182-L194](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L182-L194) |
| `init_dist` | gloo 백엔드로 `init_process_group`, 이어서 DP, EP, CP 그룹 생성 | [base.py#L315-L344](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L315-L344) |
| slave만 | `gc.freeze()` 후 RPC 루프 진입, 부모가 죽으면 SIGKILL 받도록 `prctl` | [base.py#L120-L133](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L120-L133) |

기본 프로세스 그룹이 gloo라는 점이 눈에 띈다. 전역 rank는 `engine_rank * cp_size + cp_rank`이고 초기화 주소는 `tcp://MASTER_ADDR:(MASTER_PORT - 50 + engine_group_rank)`다. GPU 통신용 그룹은 `init_parallel_groups`가 따로 만든다(10장). 3장에서 본 대로 ZOS 스케줄러의 DP 투표가 CPU 그룹을 쓰는 것과 맞물린다.

## 메모리 프로파일과 블록 수

엔진의 `_init_cache`가 부르는 `profile_num_available_blocks`는 가짜 입력으로 최대 부하 스텝을 한 번 돌려 최대 메모리를 잰다. 프로파일용 시퀀스는 `profile_run`이 만든다. prompt 엔진이면 `max_num_batched_tokens`를 `max_num_seqs`개로 나눈 길이이고 첫 시퀀스의 `seqlen_k`만 `runtime_max_model_len`으로 늘린다. decode 전용 엔진이면 시퀀스마다 `1 + nextn` 토큰이다([worker/model_runner.py#L443-L496](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/model_runner.py#L443-L496)).

### 코드 분석: profile_run 마무리

[worker/model_runner.py#L498-L512](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/model_runner.py#L498-L512)에서 임시 KV cache와 MoE 불균형 강제를 본다.

```python
        # 用一小块真实 kv cache 跑 profile，计入 cache 读写 kernel 的显存开销。
        # 分配走 cuMemAlloc，不计入 torch 显存统计，且随本函数返回即释放，因此不影响
        # profile_num_available_blocks 的计账。置零是因为 chunked-prefill 的全量 gather 会读到
        # 未写入的 cache 区域，保证读出有限值。
        # NOTE allocate_gpu_cache 会顺带写入 cache_config 的 bytes_per_block / num_state_seqs /
        # bytes_of_state_per_seqs，与之后真实分配写入的值完全相同（只有 num_gpu_blocks 不同）。
        logger.info(f"Allocating temporary kvcache for profile run: {num_blocks=}")
        cache_engine = CacheEngine(cache_config, self.model.cfg, max_num_seqs, num_gpu_blocks=num_blocks)
        cache_engine.gpu_blocks.zero_()
        cache_engine.gpu_states.zero_()

        # Run the model with dummy inputs, with MoE simulated imbalance enabled.
        envs.DEEPINFER_FORCE_IMBALANCE_RATIO = 1.8
        self.execute_model_profile_run(seqs, cache_engine.gpu_cache)
        envs.DEEPINFER_FORCE_IMBALANCE_RATIO = 0.0
```

두 가지 장치가 있다. 첫째, 프로파일 시퀀스에 꼭 맞는 작은 KV cache를 실제로 할당해 cache 읽기, 쓰기 커널의 작업 메모리까지 잡는다. 이 할당은 `cuMemAlloc`을 쓰므로 torch 통계에 섞이지 않는다고 주석이 설명한다. 둘째, MoE 게이트에 불균형 비율 1.8을 강제한다. 무작위 토큰은 전문가에 고르게 퍼지지만 실제 트래픽은 몰리므로, 전문가 쪽 버퍼의 최대치를 보수적으로 잡으려는 것이다(추정. 비율 1.8의 근거는 코드에 없다).

이어서 `profile_num_available_blocks`가 남은 메모리로 블록 수를 계산한다([worker/base.py#L228-L246](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L228-L246)).

```python
        peak_torch_allocated_memory = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        torch_allocated_memory = torch.cuda.memory_allocated()
        free_memory, _ = torch.cuda.mem_get_info()
        total_allocated_memory = total_gpu_memory - free_memory
        non_torch_allocated_memory = total_allocated_memory - torch_allocated_memory
        peak_allocated_memory = peak_torch_allocated_memory + max(non_torch_allocated_memory, 0)
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

식으로 쓰면 다음과 같다.

$$
N_{\text{blocks}} = \left\lfloor \frac{M_{\text{total}} \cdot u - (P_{\text{torch}} + M_{\text{non-torch}}) - 2 S_{\text{state}}}{B_{\text{block}}} \right\rfloor
$$

$u$는 `gpu_memory_utilization`, $P_{\text{torch}}$는 프로파일 중 torch 최대 할당량, $M_{\text{non-torch}}$는 NCCL, DeepEP 버퍼처럼 torch 밖에서 잡힌 메모리, $S_{\text{state}}$는 시퀀스별 고정 크기 state(SWA ring과 압축기 state)이며 저장용과 복사용 두 벌이다. 블록 하나의 바이트 크기와 state 모양은 5장에서 다룬다. 마지막에 world 전체 평균을 짝수로 내린 값이 모든 rank의 블록 수가 된다(3장 주의 참고).

> **주의**: CUDA graph 메모리는 이 계산에 들어가지 않는다. graph 캡처(`warm_up_model`)는 KV cache 할당 뒤에 일어나므로 $u$에서 남겨 둔 여유 안에 들어가야 한다. `LLM` docstring이 graph 사용 중 illegal memory access가 나면 `gpu_memory_utilization`을 낮추라고 한 이유다.
