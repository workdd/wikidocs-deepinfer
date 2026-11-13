플랫폼이 빌드와 실행에서 나뉘는 지점을 본다.

## 빌드 시점의 분기

### 빌드 변수와 의존성

빌드는 환경변수 `DEEPINFER_ACCELERATOR` 하나로 갈린다. 값이 `nvidia`나 `ascend`가 아니면 [`build.sh`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/build.sh#L5-L25)가 사용법을 출력하고 끝난다. README는 실행 환경에서도 이 변수를 계속 export해 두라고 적는다.

| 항목 | NVIDIA | Ascend | 근거 |
|---|---|---|---|
| PyTorch | `torch==2.13.0+cu132` | `torch==2.13.0+cpu` + `torch-npu==2.13.0rc1` | [`requirements-nvidia.txt`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/requirements-nvidia.txt#L1-L5), [`requirements-ascend.txt`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/requirements-ascend.txt#L1-L4) |
| 추가 패키지 | `flash-attn-4==4.0.0b34` (ViT attention) | 없음 | 같은 파일 |
| 통신 라이브러리 | `nvidia-nccl-cu13==2.32.3`로 강제 교체 | torch-npu 내장 HCCL | [`build.sh#L16-L19`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/build.sh#L16-L19) |
| 툴체인 | CUDA 13 | CANN 9.2 | [`README.md#L46-L52`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/README.md#L46-L52) |
| 공통 | `tilelang==0.1.15`, `tile-kernels==2.1.0`, `xgrammar==0.2.2` 등 | 같음 | [`requirements.txt`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/requirements.txt#L1-L21) |

Ascend 쪽은 CPU용 torch 휠에 `torch-npu`를 추가 설치한다. NPU 장치 지원은 전부 `torch_npu` 플러그인이 맡는다.

### vendor 서브모듈

| 서브모듈 | 기준 커밋의 gitlink | NVIDIA 빌드 | Ascend 빌드 | 설치 이름 |
|---|---|---|---|---|
| `vendor/DeepSelect` | `bfa4507` | 빌드 (`DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA`) | 빌드 (`=ASCEND`) | `deep_select` |
| `vendor/FlashMLA` | `2e5429f` | 빌드 (`FLASH_MLA_BUILD_TARGET_PLATFORM=CUDA`) | 빌드 (`=ASCEND`) | `flash_mla` |
| `vendor/DeepEP` | `f6aa38f` | 빌드 | 사용 안 함 | `deep_ep` |
| `vendor/DeepEP-Ascend` | `3b25377` | 사용 안 함 | 빌드 | `deep_ep` |
| `vendor/DeepGEMM` | `1fc9c66` | 빌드 (`DG_FORCE_BUILD=1`) | 사용 안 함 | `deep_gemm` |
| `vendor/DeepGEMM-Ascend` | `3a35326` | 사용 안 함 | 빌드 | `deep_gemm` |

[`scripts/build_vendor.sh#L7-L22`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/scripts/build_vendor.sh#L7-L22)가 이 표를 그대로 코드로 적는다.

```bash
case "${DEEPINFER_ACCELERATOR:-nvidia}" in
  nvidia)
    BUILD_TARGET=CUDA
    PACKAGES=(DeepSelect FlashMLA DeepEP DeepGEMM)
    INSTALL=(deep_select flash_mla deep_ep deep_gemm)
    ;;
  ascend)
    BUILD_TARGET=ASCEND
    PACKAGES=(DeepSelect FlashMLA DeepEP-Ascend DeepGEMM-Ascend)
    INSTALL=(deep_select flash_mla deep_ep deep_gemm)
    ;;
```

`INSTALL` 배열이 두 경우에 똑같다는 점이 핵심이다. DeepSelect와 FlashMLA는 한 저장소가 빌드 대상 변수로 두 플랫폼을 다 만들고, DeepEP와 DeepGEMM은 Ascend용 저장소를 따로 두되 패키지 이름과 Python API를 맞췄다. 그래서 7장의 `deep_gemm.fp8_fp4_mega_moe`나 `deep_ep.EPBuffer` 호출이 플랫폼 분기 없이 그대로 쓰인다. 두 구현이 모든 함수를 똑같이 지원하는지는 서브모듈 코드를 읽지 않아 확인하지 못했다(미검증). 아래에서 보듯 모델 코드가 NPU에서 일부 기능을 끄는 것으로 보아 지원 범위에 차이가 있다.

휠은 `build/<가속기>-<arch>/vendor-wheels`에 모았다가 `--no-index`로 설치한다. 서브모듈의 서브모듈까지 `git submodule update --init --recursive`로 받는다.

### 확장 모듈 `_C`

[`setup.py`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/setup.py#L38-L72)는 같은 C++ 스케줄러 소스(`csrc/engine/*`)를 공통으로 두고, 장치 보조 파일 두 개만 교체한다.

| 항목 | NVIDIA (`CUDAExtension`) | Ascend (`NpuExtension`) |
|---|---|---|
| 장치 보조 소스 | `csrc/cuda_utils.cpp`, `csrc/graph_timer/graph_timer_cuda.cpp` | `csrc/asc_utils.cpp`, `csrc/graph_timer/graph_timer_asc.cpp` |
| 컴파일 플래그 | `-fvisibility=hidden` | `-fopenmp -fvisibility=hidden -DWITH_NPU` |
| 링크 | `libcuda` (`CUDA_HOME/lib64/stubs`) | torch_npu 기본 |
| 빌드 조건 | `CONDA_PREFIX` 또는 `VIRTUAL_ENV` 필수 | 없음 |

두 보조 파일은 같은 이름의 함수(`import_ipc_tensor`, `export_ipc_tensor`, `alloc_tensor`, `cuda_copy_async`, `get_current_stream`)를 각자의 런타임 API로 구현한다([`cuda_utils.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/cuda_utils.cpp#L34-L96), [`asc_utils.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/csrc/asc_utils.cpp#L32-L106)). Python에서 보면 함수 이름이 같으므로 호출부는 하나다. NPU용 `get_current_stream`은 `c10_npu::getCurrentNPUStream()`을 쓴다.

빌드 산출물은 `deepinfer/_so/<가속기>-<arch>/`에 링크된다([`scripts/build.sh#L11-L28`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/scripts/build.sh#L11-L28)). 패키지 초기화가 이 폴더를 `__path__`에 추가한다.

무엇을 보는지: torch를 import하지 않고 플랫폼 태그를 정해 `.so` 폴더를 고르는 부분. [`deepinfer/__init__.py#L9-L32`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/__init__.py#L9-L32)

```python
def _so_platform_tag() -> str:
    accelerator = envs.DEEPINFER_ACCELERATOR
    if not accelerator:
        try:
            from deep_cluster.hardware import accelerator_kind

            accelerator = accelerator_kind()
        except Exception:
            accelerator = ""
    if accelerator not in ("nvidia", "ascend"):
        accelerator = "nvidia"
    return f"{accelerator}-{os.uname().machine}"


_so_dir = os.path.join(os.path.dirname(__file__), "_so", _so_platform_tag())
if os.path.isdir(_so_dir):
    __path__.append(_so_dir)
```

(docstring을 줄였다.)

주석에 따르면 이 방식 덕분에 같은 arch의 NVIDIA 빌드와 Ascend 빌드가 한 소스 트리에 함께 있을 수 있다. `DEEPINFER_ACCELERATOR`의 기본값은 `"nvidia"`라([`environ.py#L108`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/environ.py#L108)) 변수가 비어 있지 않은 한 탐지 경로는 타지 않는다. Ascend 장비에서 이 변수를 export하지 않으면 NVIDIA용 `.so` 폴더를 찾게 된다. README가 변수를 계속 export하라고 한 이유다.

## 실행 시점의 분기

### 가속기 탐지

`deep_cluster.hardware`는 torch, CUDA, ACL 어느 것도 import하지 않고 장치 파일만 본다.

무엇을 보는지: 탐지 순서. [`deep_cluster/hardware/device.py#L17-L23`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deep_cluster/hardware/device.py#L17-L23)

```python
@functools.cache
def accelerator_kind(*, _root: Path = Path("/")) -> AcceleratorKind:
    if ascend.is_available(_root=_root):
        return "ascend"
    if nvidia.is_available(_root=_root):
        return "nvidia"
    return "cpu"
```

| 항목 | NVIDIA | Ascend |
|---|---|---|
| 존재 판정 | `/proc/driver/nvidia/gpus/*` 또는 `/dev/nvidiaN` | `/dev/davinci_manager`, `/dev/davinciN`, 또는 `/sys/class/ubcore/udma*`의 `ub_totalues > 0` |
| 장치 목록 | procfs의 PCI 주소 | ubcore 항목, 없으면 davinci 장치 번호 |
| NUMA 노드 | `/sys/bus/pci/devices/<bus>/numa_node` | `npu-smi info -t topo`의 CPU 친화도와 `lscpu`를 대조, 실패하면 ubcore `numa` 값으로 추정 |
| 보이는 장치 변수 | `CUDA_VISIBLE_DEVICES` | `ASCEND_RT_VISIBLE_DEVICES` |
| 세대 판정 | NVML로 compute capability 조회 (`is_sm_major_10`) | 없음 |

Ascend NUMA 탐지([`ascend.py#L107-L144`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deep_cluster/hardware/ascend.py#L107-L144))는 `npu-smi` 출력에서 장치별 CPU 친화도 목록을 읽고, 물리 코어 집합과 가장 많이 겹치는 NUMA 노드를 고른다. 이 경로가 실패하면 ubcore가 알려 주는 값에서 `(raw // 2) * 2 + index % 2`로 계산한다. 왜 이 공식인지는 코드에 설명이 없다(미검증).

`is_sm_major_10`은 NVML(`libnvidia-ml`)을 ctypes로 불러 CUDA 컨텍스트를 만들지 않고 compute capability를 읽는다([`nvidia.py#L111-L129`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deep_cluster/hardware/nvidia.py#L111-L129)). 이 값은 통계 로거의 이론 성능 상한에 쓰인다. [`get_bf16_tflops_limit`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deep_cluster/hardware/device.py#L151-L156)은 Ascend면 432, SM100이면 aarch64에서 2500, 그 밖에서 2250을 돌려주고 다른 GPU면 예외를 던진다. 엔진은 이 값에 2를 곱해 FP8 기준 상한으로 쓴다([`llm_engine.py#L304`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/engine/llm_engine.py#L304)). 공식 지원 하드웨어가 SM100과 Ascend 950뿐이라는 README의 말이 코드에 이렇게 박혀 있다. 이 TFLOPS 값은 코드 상수를 옮긴 것이며 제조사 사양과 대조하지 않았다.

### `Platform` 객체

`deepinfer.PLATFORM`은 모듈 수준 `__getattr__`으로 처음 접근할 때 만든다([`deepinfer/__init__.py#L53-L64`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/__init__.py#L53-L64)). 파일 첫 줄 주석이 "이 파일에서 torch를 import하거나 장치를 초기화하지 말 것, 하위 프로세스가 고장 난다"고 적는다. NPU 플랫폼 초기화는 `torch.cuda`를 패치하고 ACL 런타임을 띄우므로, 보이는 장치 변수를 정한 뒤에 일어나야 한다.

| 속성 | `PlatformGPU` | `PlatformNPU` |
|---|---|---|
| `tilelang_ops_backend` | `"cuda"` | `"asc"` |
| `distributed_backend` | `"nccl"` | `"hccl"` |
| `fp8_sf_type` | `torch.int32` | `torch.int16` |
| `get_distributed_options` | NCCL `max_ctas = min_ctas = 32` | HCCL 기본 옵션 |
| `pin_host_memory` | `cudaHostRegister` | `aclrtHostRegisterV2(..., PINNED)` |

`fp8_sf_type`은 ue8m0 scale(지수만 담은 1바이트) 여러 개를 묶어 담는 정수 타입이다. CUDA는 4개를 `int32` 하나에, Ascend는 2개를 `int16` 하나에 묶는다([`fused_transform.py#L61-L66`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/model_executor/layers/tilelang_ops/fused_transform.py#L61-L66)). KV 캐시, Engram, 선형층 등 scale을 다루는 곳마다 `PLATFORM.fp8_sf_type.itemsize`로 묶음 크기를 계산한다. 그래서 KV 캐시의 바이트 배치도 플랫폼마다 다르다. 같은 체크포인트라도 한 플랫폼에서 만든 KV 캐시를 다른 플랫폼에서 그대로 읽을 수는 없을 것으로 보인다(추정).

### 코드 분석: NPU 플랫폼 초기화

무엇을 보는지: `PlatformNPU`가 torch_npu를 불러 CUDA 스타일 코드가 NPU에서 돌게 만드는 부분. [`platform.py#L91-L143`](https://github.com/deepseek-ai/DeepInfer/blob/a679bb86ce36e883a98a97733492b686e5b9a08e/deepinfer/platform.py#L91-L143)

```python
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TASK_QUEUE_ENABLE"] = "0"
os.environ["HCCL_BUFFSIZE"] = "16"
os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "240")
os.environ.setdefault("HCCL_EXEC_TIMEOUT", "200")
os.environ.setdefault("TORCH_NPU_DEVICE_CAPABILITY", "9.0")

import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

os.environ.setdefault("TILELANG_DEFAULT_TARGET", "ascend")

torch_npu.npu.config.allow_internal_format = True
torch_npu.npu.set_compile_mode(jit_compile=False)

import torch
torch.npu.Stream.cuda_stream = property(lambda self: self.npu_stream)

self.fp8_sf_type = torch.int16

import deepinfer.npu_ops  # noqa: F401
```

(장치 변수 검사, 포트 범위 설정, 주석을 줄였다.)

이 몇 줄이 이원화의 바탕이다.

- `transfer_to_npu`는 `torch.cuda.*` 호출을 `torch.npu.*`로 돌린다. 그래서 모델 코드 전체에 남아 있는 `device="cuda"`, `torch.cuda.CUDAGraph`, `torch.cuda.Event`가 NPU에서도 동작한다
- 그 부작용으로 `torch.cuda.is_available()`이 NPU 장비에서 참이 된다. TileLang이 이 값을 보고 CUDA 타깃으로 컴파일하지 않도록 `TILELANG_DEFAULT_TARGET=ascend`를 미리 박는다
- `.cuda_stream` 속성을 읽는 코드를 위해 NPU 스트림에 같은 이름의 속성을 붙인다
- `TORCH_NPU_DEVICE_CAPABILITY=9.0`은 torch_npu 버그를 피하는 임시 조치라고 주석(`TODO(NPU)`)에 적혀 있다

생성자는 `CUDA_VISIBLE_DEVICES`와 `ASCEND_RT_VISIBLE_DEVICES`가 다르면 예외를 던지고, 앞쪽이 비어 있으면 뒤쪽 값으로 채운다. 예제 스크립트와 테스트 conftest가 두 변수를 항상 같이 맞추는 이유다.
