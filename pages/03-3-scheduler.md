스케줄링을 맡는 C++ 코드와 GPU 실행과 겹쳐 도는 비동기 구조를 본다.

## C++ 스케줄러

### 코드 분석: NORMAL 엔진의 prefill 투표

NORMAL 엔진은 한 스텝에 prefill과 decode를 섞지 않는다. 스텝 전체가 prompt 스텝이거나 decode 스텝이고, 모든 DP rank가 같은 종류여야 한다. [csrc/engine/scheduler.hpp#L308-L335](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L308-L335)의 투표 로직을 본다.

```cpp
            bool is_any_point = (decode_step != 0) && (decode_step % prefill_any_attempt_interval == 0);
            bool is_all_point = (decode_step % prefill_attempt_interval == 0);
            if (!is_any_point && !is_all_point) {
                return _schedule_decode(acquire);
            }
            bool want = (get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK) && _has_prefill_waiting();
            bool do_prefill;
            if (is_any_point) {
                do_prefill = (_dp_allreduce(want ? 0 : 1, c10d::ReduceOp::MIN) == 0); // ANY：任一 rank 想做即触发
            } else {
                do_prefill = (_dp_allreduce(want ? 1 : 0, c10d::ReduceOp::MIN) == 1); // ALL：全员想做才触发
            }
            if (!do_prefill) {
                return _schedule_decode(acquire);
            }
            std::unique_ptr<SchedulerOutputs> outputs;
            if (want) {
                outputs = _schedule_prefill(acquire);
            }

            bool any_scheduled = (_dp_allreduce(outputs ? 0 : 1, c10d::ReduceOp::MIN) == 0);
            if (!any_scheduled) {
                return _schedule_decode(acquire);
            }
```

`decode_step`은 마지막 prompt 스텝 이후 decode 스텝 수다. 투표 지점은 두 종류다.

| 지점 | 주기 (기본값) | 조건 | 의도 |
| --- | --- | --- | --- |
| ALL | `prefill_attempt_interval` decode 스텝마다 (70, `decode_step=0` 포함) | 모든 rank가 prefill을 원할 때 | 한 rank만 일하고 나머지가 빈 prompt 스텝을 따라 도는 낭비 방지 |
| ANY | `prefill_any_attempt_interval` decode 스텝마다 (100) | 한 rank라도 원할 때 | 혼자 prefill이 쌓인 rank가 계속 밀리는 것 방지. 최대 100 decode 스텝 대기 |

기본값은 [deepinfer/environ.py#L181-L184](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L181-L184)의 `DEEPINFER_PREFILL_ATTEMPT_INTERVAL=70`, `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=100`이다. "원한다"는 KV 사용률이 `GPU_CACHE_PREFILL_WATERMARK`(0.9) 이하이고 waiting 맨 앞 시퀀스가 스케줄 가능하다는 뜻이다. 실제로 prefill을 짜 봤는데 아무 rank도 성공하지 못하면 두 번째 all-reduce가 decode로 돌린다. 코드 주석은 이 두 번째 투표가 없으면 running이 꽉 차고 waiting이 남은 상태에서 빈 prompt 스텝이 `decode_step`을 계속 0으로 되돌려 무한 루프에 빠진다고 설명한다. 자기 rank에는 prefill이 없지만 다른 rank가 하는 경우, 빈 `SchedulerOutputs(prompt_run=true)`를 만들어 스텝 종류를 맞춘다.

prompt 스텝이 끝나면 `decode_step`이 0이 되고 0은 ALL 지점이다. 그래서 prefill을 연달아 하려면 매번 모든 rank가 원해야 한다. 정리하면 prefill은 기본적으로 70 decode 스텝에 한 번 열리는 창이다. vLLM의 chunked prefill이 decode와 prefill을 한 배치에 섞는 방식과 반대 방향이다(vLLM 쪽은 미검증). decode 지연을 일정하게 유지하고 prefill은 일괄 처리하는 대규모 DP 배치 운영을 겨냥한 설계로 보인다(추정).

### prefill 스케줄의 제약

`_schedule_prefill`은 waiting 맨 앞부터 순서대로 꺼내며 제약을 하나씩 검사한다([csrc/engine/scheduler.hpp#L351-L554](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L351-L554)). 맨 앞이 걸리면 뒤를 보지 않고 멈춘다(FCFS).

| 검사 | 걸렸을 때 |
| --- | --- |
| 이미 종료 상태(abort, VE 실패, SO 실패) | `ignored_seqs`로 빼고 계속 |
| `schedulable=false` (멀티모달 전처리 미완료 등) | 멈춤 |
| 길이가 `prompt_limit` 초과 | `FINISHED_IGNORED` |
| 블록 또는 state 슬롯 할당 `LATER` | 멈춤. `NEVER`면 `FINISHED_IGNORED` |
| running 수가 `max_num_seqs`에 도달 | 멈춤 |
| prefill 시퀀스 길이 합이 `prefill_max_sum_seqlen_k` 초과 | 경고 후 멈춤 |
| 남은 토큰 예산보다 길면 | 예산만큼 자른 청크로 스케줄(CP 정렬 단위 `cp_size*2`로 내림) |
| 청크 끝이 이미지 토큰 중간 | 이미지 단위 경계로 줄이거나(뒤 시퀀스) 늘림(첫 시퀀스) |
| 잘렸는데 `chunked_prefill=false` | 멈춤 |
| 청크가 `max(lookahead_tokens, 1)`보다 짧음 | 멈춤 |

`prompt_limit`은 chunked prefill이 켜져 있으면 `max_model_len`, 꺼져 있으면 `min(max_model_len, max_num_batched_tokens)`이다([csrc/engine/scheduler.hpp#L112](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L112)). NORMAL 엔진은 청크로 잘린 시퀀스를 넣은 순간 루프를 끝낸다([csrc/engine/scheduler.hpp#L543-L545](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L543-L545)). 잘린 시퀀스는 상태를 `WAITING`으로 유지한 채 `process_outputs`에서 waiting 맨 앞에 되돌아가고, 다음 prompt 스텝에서 이어서 처리된다. `chunk_checkpoints`가 지정된 요청은 그 위치에서 청크를 끊는다. prefix 저장 지점을 맞추기 위한 장치로 보인다(추정).

### 코드 분석: decode 스케줄과 선점

[csrc/engine/scheduler.hpp#L622-L646](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L622-L646)에서 decode 배치를 고르는 부분을 본다.

```cpp
        // LCFS
        std::vector<SequencePtr> scheduled;
        std::sort(running.begin(), running.end(),
            [](const SequencePtr &a, const SequencePtr &b) { return a->get_output_len() > b->get_output_len(); });
        scheduled.reserve(max_seqs);
        size_t l = 0, r = running.size();
        while (l < r && scheduled.size() < max_seqs) {
            auto &seq = running[r - 1];
            while (!block_manager->can_allocate_single_nolock() && r - l > 1) {
                auto victim_seq = running[l];
                _preempt(victim_seq, ignored_seqs);
                l += 1;
            }
            if (block_manager->can_allocate_single_nolock()) {
                _append_slot(*seq);
                scheduled.push_back(std::move(seq));
            } else {
                _preempt(seq, ignored_seqs);
            }

            r -= 1;
        }

        TORCH_CHECK(l == r, "decode 时 running 有排队，已经调度 ", scheduled.size(), "个，还剩 ", r - l, " 个，不应该");
        running.clear();
```

running을 출력 길이 내림차순으로 정렬한 뒤, 뒤쪽(출력이 짧은 것)부터 블록을 붙여 스케줄하고, 블록이 모자라면 앞쪽(출력이 긴 것)부터 선점한다. 주석의 LCFS와 함수 docstring의 "output 最短의 seq를 선점"은 서로 다르게 읽힌다. 코드상 희생자는 `running[l]`, 즉 정렬 후 맨 앞인 출력이 가장 긴 시퀀스다. 코드 동작을 기준으로 삼는다. 스케줄 후에는 다시 전체 길이 내림차순으로 정렬하고, MTP 가변 길이 모듈(`_mtp_varlen.schedule`)이 시퀀스별 `seqlen_q`를 정한다(8장).

선점은 `_preempt`가 처리한다([csrc/engine/scheduler.hpp#L766-L790](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L766-L790)). `stop_preempted_request=true`이거나 구조화 출력이 진행 중이면 `FINISHED_PREEMPTED`로 끝내고, 아니면 KV 진행 상황을 0으로 되돌려 waiting 맨 앞에 넣어 재계산한다. 스왑(CPU로 KV 내리기)은 없다. COMPLETION 엔진은 prefill 경로가 없어 재계산이 불가능하므로 `stop_preempted_request=false` 조합을 생성자에서 막는다([csrc/engine/scheduler.hpp#L101-L104](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L101-L104)).

### 코드 분석: schedule()의 메타데이터 생성 방식

[csrc/engine/scheduler.hpp#L692-L707](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L692-L707)에서 시퀀스마다 `seqlen_k`를 정하는 부분을 본다.

```cpp
        for (auto &&p_seq : scheduler_outputs.scheduled_seqs) {
            auto &&seq = p_seq.get();
            seq->mtp.schedule_cnt += 1;
            auto golden_token = seq->mtp.next_golden_tokens(seq->get_output_len(), nextn);
            int seqlen_q = scheduler_outputs.seqlens_q[i];
            int seqlen_k = seq->cache.num_tokens + seq->schedule.total_inflight_tokens + seqlen_q;
            int position_end = scheduler_outputs.prompt_run ? seqlen_k : seqlen_k - seqlen_q + max_seqlen_q + nextn;
            TORCH_CHECK(position_end <= runtime_max_model_len,
                std::format("position_end={} exceeds runtime_max_model_len={}: request_id={} branch={} len={} num_tokens={} inflight={} "
                            "seqlen_q={} prompt_run={}",
                    position_end, runtime_max_model_len, seq->request.request_id, seq->schedule.branch_id, seq->get_len(),
                    seq->cache.num_tokens, seq->schedule.total_inflight_tokens, seqlen_q, scheduler_outputs.prompt_run));
            auto &&p_metadata =
                sequence_metadata_pool->allocate(seq, scheduler_outputs.prompt_run, seqlen_q, seqlen_k, std::move(golden_token));
```

`seqlen_k`에 `total_inflight_tokens`가 더해진다는 점이 비동기 스케줄링의 핵심이다. 앞 스텝이 아직 GPU에서 도는 중이라 `cache.num_tokens`에는 그 스텝 몫이 반영되지 않았다. 스케줄러는 앞 스텝이 최대 `nextn + 1`개 토큰을 만들 것으로 가정하고(`start_inflight_step(max_seqlen_q)`), 실제 수용 개수는 GPU에서 보정한다(4장 `_correct_seqlens_qk`). 이 시점에 시퀀스마다 FLOPs 추정치도 더해 `attn_tflops`, `total_tflops`로 내보낸다. `StatsLogger`의 MFU가 이 값을 쓴다.

### 코드 분석: process_outputs의 낙관적 전진

결과 반영은 `process_outputs`가 한다. [csrc/engine/scheduler.hpp#L927-L940](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L927-L940)을 본다.

```cpp
            seq.cache.advance_optimistically(scheduler_outputs.seqlens_q[i]);
            if (so_status_bool)
                seq.structured_output.status = so_status[i].cast<int>();

            // 乐观推进后 cache.num_tokens 的不变式（input.tokens 只含已提交 token，不含 draft，契约见其声明处；
            // 在途被抢占的 seq 已在上面跳过）：
            //   prompt 步：seqlen_q 只覆盖 prompt 本身，推进后 <= len，等于 len 说明本 seq 的 prefill 已完成；
            //   decode 步：seqlen_q = 1 + 参与验证的 draft 数，推进后其值落在 [len, len + nextn] 内，
            //   多算的部分随后按实际接受的 token 数（token_cnt）回填修正。
            TORCH_CHECK(seq.cache.num_tokens <= seq.input.tokens.get_len() + seq.get_nextn_len());
            bool prefill_finish_or_is_decode = (seq.cache.num_tokens >= seq.input.tokens.get_len());
            TORCH_CHECK(scheduler_outputs.prompt_run || prefill_finish_or_is_decode, "decode 步推进后必然 >= len，不应该：",
                seq.cache.num_tokens, " < ", seq.input.tokens.get_len());
```

KV에 쓰인 토큰 수를 일단 `seqlen_q`만큼 앞으로 민 다음, 수용된 draft와 새 토큰을 붙이고 나서 `commit_processed_tokens(token_cnt)`로 실제 수용 개수에 맞춰 되돌린다([csrc/engine/sequence.hpp#L216-L221](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/sequence.hpp#L216-L221)). 검증에서 탈락한 draft 자리는 KV를 차지하지 않은 것으로 처리된다. 앞 단계에서 `finish_inflight_step()`이 이번 스텝 몫의 inflight 토큰을 지운다. 비동기 실행 중 이미 선점되어 waiting으로 돌아간 시퀀스는 결과를 버리고 건너뛴다([csrc/engine/scheduler.hpp#L917-L925](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/engine/scheduler.hpp#L917-L925)).

## 비동기 스케줄링 (ZOS)

ZOS는 CPU 스케줄링, 결과 처리와 GPU 실행을 겹치는 설정이다. 기본값은 `EngineType.default_zero_overhead_scheduling`이 정하고, NORMAL과 COMPLETION은 켜짐, PROMPT는 꺼짐이다([engine/engine_type.py#L4-L17](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/engine_type.py#L4-L17)). 켜고 끄는 것에 따라 바뀌는 지점은 흩어져 있으므로 표로 모은다.

| 지점 | ZOS 켜짐 | ZOS 꺼짐 |
| --- | --- | --- |
| 스케줄러 DP 투표 그룹 | CPU(gloo) DP 그룹 | GPU DP 그룹 ([llm_engine.py#L313](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L313)) |
| `has_unfinished_requests` all-reduce | CPU 텐서 | CUDA 텐서 |
| decode 결과 후처리 | 다음 스텝 forward 발행 직후 | 같은 스텝 샘플링 직후 |
| `runtime_max_model_len` 여유 | `nextn + 1` 추가 | 추가 없음 |
| decode 입력 준비 | GPU에서 앞 스텝 샘플 결과로 보정 (`prepare_decode_v4_zos`) | CPU에서 C++로 준비 (`prepare_decode_v4`) |
| CUDA graph 입력 | 장치 버퍼에 직접 기록 (`input_on_device`) | CPU 버퍼에서 graph 안 복사 |
| `step()` 반환 | 앞 스텝 출력 | 이번 스텝 출력 |

GPU 그룹으로 all-reduce하면 GPU 스트림에 동기점이 생기므로, 앞 스텝 실행과 겹치려면 CPU 그룹을 써야 한다(추정). 4장에서 볼 워커의 `execute_model_step`이 "이번 스텝 forward 발행, 앞 스텝 후처리, 이번 스텝 샘플링 발행" 순서로 도는 것이 이 겹침의 실체다.

### 코드 분석: DeferredRelease

후처리가 한 스텝 밀리면 자원 해제도 밀려야 한다. [engine/llm_engine.py#L100-L119](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L100-L119)를 본다.

```python
    def release(self) -> None:
        """在 pre_sampler_sync 调用，释放资源"""
        if self.sequence_metadata_to_release:
            self.sequence_metadata_pool.release(self.sequence_metadata_to_release)
            self.sequence_metadata_to_release = []

        self.request_output_pool.release(self.request_outputs_to_release)
        self.request_outputs_to_release = np.empty(0, dtype=np.uint64)

        if self.clone_unpins_to_release is not None:
            states_to_unpin, blocks_to_unpin = self.clone_unpins_to_release
            self.clone_unpins_to_release = None
            self.clone_done_evt.synchronize()
            if states_to_unpin.size > 0:
                self.state_manager.unpin_blocks(states_to_unpin)
            if blocks_to_unpin.size > 0:
                self.block_manager.unpin_blocks(blocks_to_unpin)

        # 释放 multi modal handles
        free_deferred_pool_handles()
```

`release()`는 `pre_sampler_sync`에서 불린다. 샘플링 커널을 발행한 직후, GPU가 샘플링을 도는 동안 CPU가 해제를 처리한다. 해제 대상은 세 종류다. 앞 스텝의 `SequenceMetadata`(워커가 후처리에서 참조하므로 그 전에 해제하면 안 됨), 앞 스텝 `step()`이 돌려준 `RequestOutput`(호출자가 읽을 시간), fork 복사 원본 블록의 pin. 객체를 매번 만들지 않고 풀로 재사용하는 것은 스텝당 파이썬 할당을 줄이려는 의도로 읽힌다(추정). 워커도 생성자에서 `gc.disable()`을 하고 `DEEPINFER_GC_INTERVAL_STEP`(기본 1000) 스텝마다 수동으로 `gc.collect()`를 부른다([worker/base.py#L102-L106](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L102-L106), [worker/base.py#L153-L157](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/worker/base.py#L153-L157)).

### 코드 분석: post_process 콜백

워커가 후처리 시점에 부르는 엔진 콜백은 [engine/llm_engine.py#L984-L1015](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L984-L1015)다.

```python
    def post_process(self, step_state: WorkerStepState) -> np.ndarray[np.uint64]:
        seq_metadata_list = step_state.seq_metadata_list
        scheduler_outputs = step_state.scheduler_outputs
        output = step_state.sampler_output if not step_state.is_placeholder else None
        if scheduler_outputs.prompt_run:
            self.pe_do_save_kvcache(scheduler_outputs.scheduled_seqs, scheduler_outputs.finished_seqs)

        with cpu_timer.record("bubble"):
            request_outputs, log_stats_func = self._process_model_outputs(output, scheduler_outputs)
        self.post_process_model_outputs()

        if step_state.forward_events is not None and step_state.sample_events is not None:
            # Worker has synchronized this step's sample_end_evt before calling post_process.
            self.scheduler.update_mtp_runtime(
                len(seq_metadata_list),
                step_state.graph_size,
                step_state.forward_events[0].elapsed_time(step_state.forward_events[1]),
                step_state.sample_events[0].elapsed_time(step_state.sample_events[1]),
            )

        try:
            log_stats_func()
        except Exception as e:
            traceback.print_exc()
```

`_process_model_outputs`는 MTP 정보 갱신(`process_MTPInfo`), 스케줄러 상태 반영(`process_outputs`), 종료 집계, C++ `build_request_outputs` 호출을 묶는다. cpu_timer 이름이 `bubble`인 것은 ZOS에서도 이 구간이 GPU를 기다리게 할 수 있는 부분이라는 표시로 읽힌다(추정). MTP 가변 길이가 켜져 있으면 측정한 forward, sampler 시간을 스케줄러의 시간 모델에 넣는다. 8장에서 다룬다. 이름에 `pe_`, `do_save_kvcache`가 붙은 가상 함수들은 PD 분리와 KV 저장용 확장 지점으로, 기반 클래스에서는 비어 있다.
