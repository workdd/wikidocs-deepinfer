토큰 샘플링, 구조화 출력, 여러 GPU 에 걸친 실행과 P/D 분리를 다룬다.

| 장 | 핵심 |
|---|---|
| 09. 샘플링과 구조화 출력 | 모델이 hidden state를 내놓은 뒤 토큰 하나를 고르기까지의 과정은 `Sampler` 한 클래스가 맡는다. |
| 10. 분산 실행과 prefill/decode 분리 | DeepInfer의 분산 실행은 "attention은 data parallel, MoE는 expert parallel"이라는 한 문장으로 요약된다. |
