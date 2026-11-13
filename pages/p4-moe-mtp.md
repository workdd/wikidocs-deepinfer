MoE expert 병렬과 MTP 추측 디코딩을 다룬다.

| 장 | 핵심 |
|---|---|
| 07. MoE와 Expert 병렬, EPLB | DeepSeek-V4.1-Flash의 FFN은 전부 MoE이고, DeepInfer는 이 MoE를 expert parallel(EP)로만 나눈다. |
| 08. MTP 추측 디코딩 | DeepInfer의 추측 디코딩은 코드에서 "MTP"라고 부르지만, 공개본이 지원하는 draft head는 DFlash 하나뿐이다. |
