요청을 받아 스텝 단위로 GPU 에 넘기는 엔진과 워커를 다룬다.

| 장 | 핵심 |
|---|---|
| 03. 엔진 루프: LLMEngine과 스케줄링 | DeepInfer의 엔진 루프는 파이썬 `LLMEngine.step()`이 C++ `Scheduler`에게 이번 스텝의 배치를 받아 driver 워커에 넘기는 구조이고, 비동기 스케줄링(코드 이름 ZOS, zero-overhead scheduling)이 켜져 있으면 스텝 N의 결과 처리가 스텝 N+1의 GPU 실행과 겹친다. |
| 04. 워커와 모델 러너: GPU 스텝 실행 과정 | 엔진이 넘긴 한 스텝은 워커의 `execute_model_step` 안에서 "이번 스텝 forward 발행, 앞 스텝 후처리, 이번 스텝 샘플링 발행" 순서로 돌고, 그 사이 CPU는 GPU를 기다리지 않도록 동기점을 이벤트 몇 개로 묶어 둔다. |
