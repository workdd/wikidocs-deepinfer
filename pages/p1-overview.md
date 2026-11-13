DeepInfer 의 전체 구성과 설치, 첫 실행을 다룬다.

| 장 | 핵심 |
|---|---|
| 01. DeepInfer 개요: 구성과 vLLM 계보 | DeepInfer는 Python 엔진에 C++ 스케줄러를 결합하고, 모델 실행은 DeepSeek 자체 커널(FlashMLA, DeepGEMM, DeepEP)과 TileLang 커널로 채운 구조다. |
| 02. 설치와 첫 오프라인 추론 | DeepInfer 설치는 `DEEPINFER_ACCELERATOR` 하나로 갈리는 세 단계 빌드(파이썬 의존성, vendor 연산자 휠, `deepinfer._C` 확장)로 끝나고, 첫 추론은 `examples/offline_inference.py` 한 줄로 돈다. |
