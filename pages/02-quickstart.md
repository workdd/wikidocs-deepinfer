DeepInfer 설치는 `DEEPINFER_ACCELERATOR` 하나로 갈리는 세 단계 빌드(파이썬 의존성, vendor 연산자 휠, `deepinfer._C` 확장)로 끝나고, 첫 추론은 `examples/offline_inference.py` 한 줄로 돈다. 이 장은 빌드 스크립트가 무엇을 어떤 순서로 만드는지, 예제 스크립트가 데이터 병렬 프로세스를 띄워 `LLM` 객체를 만들기까지 어떤 환경 변수와 인자를 거치는지를 코드로 따라간다.

## 준비물과 지원 범위

README가 밝히는 요구 사항은 짧다. 빈 Python 3.12 환경을 만들고 활성화한 다음 빌드하라고 적혀 있다([README.md#L46-L63](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L46-L63)).

| 항목 | 요구값 | 근거 |
| --- | --- | --- |
| OS | Linux | README Prerequisites |
| 컴파일러 | GCC 13.0 이상 | README Prerequisites |
| Python | 3.12 (`requires-python = ">=3.12"`) | [pyproject.toml#L16](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/pyproject.toml#L16) |
| NVIDIA | CUDA 13, SM100 GPU | README Prerequisites, Supported hardware |
| Ascend | CANN 9.2, Ascend 950 NPU | README Prerequisites, Supported hardware |
| 가상환경 | conda 또는 virtualenv 필수 (NVIDIA 빌드) | [setup.py#L55-L57](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/setup.py#L55-L57) |
| 지원 모델 | DeepSeek-V4.1-Flash | README Supported models |

README는 이 공개판이 오프라인 추론만 제공하고, 완전한 서비스에 필요한 KV cache 시스템, VE/Prefill/Decode 분리, DP 복제본 간 로드 밸런싱은 포함하지 않는다고 명시한다([README.md#L38-L42](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L38-L42)). 그래서 이 장의 "첫 실행"도 HTTP 서버가 아니라 파이썬 스크립트 안에서 `LLM.generate()`를 부르는 형태다.

```bash
# README 기준 설치와 첫 실행
export DEEPINFER_ACCELERATOR=nvidia   # Ascend 는 ascend
./build.sh
python examples/offline_inference.py
```

## 정리

- 설치는 `DEEPINFER_ACCELERATOR`를 export하고 `./build.sh`를 돌리면 끝난다. 내부에서 의존성 설치, vendor 서브모듈 휠 빌드, `deepinfer._C` 빌드의 세 단계가 순서대로 실행되고 Makefile 타깃으로 개별 재실행할 수 있다.
- vendor 단계는 가속기에 따라 DeepEP와 DeepGEMM의 저장소를 바꿔 끼우되 설치 패키지 이름은 같게 유지한다. 엔진 코드는 어느 구현인지 모른 채 같은 이름으로 import한다.
- `deepinfer._C`는 `.cu` 없이 `.cpp`만으로 된 CPU 쪽 확장(스케줄러, 입력 준비, graph 타이머)이다. 산출물은 `deepinfer/_so/<가속기>-<arch>/`에 링크되어 두 가속기 빌드가 한 소스 트리에 공존한다.
- 예제는 노드별로 DP rank마다 `spawn` 프로세스를 만들고, 각 프로세스가 환경 변수와 `CUDA_VISIBLE_DEVICES`를 먼저 정한 뒤 deepinfer를 import해 `LLM`을 만든다. 이 순서가 깨지면 장치 배정이 무시된다.
- 엔진 인자는 `EngineArgs`, `LLM`, `create_llm` 세 겹의 기본값을 가진다. offline 예제는 `max_num_seqs=4`, 길이 32768, `chunked_prefill=True`, `stop_preempted_request=False`로 실행한다.
- `LLM.generate()`는 `step()`을 반복한 뒤 `flush_inflight_reqs()`로 마무리하고, 출력은 다음 스텝에 회수되므로 복사해 둔다. 이 두 동작은 3장의 비동기 스케줄링에서 이어진다.
