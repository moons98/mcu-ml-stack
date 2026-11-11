# Report

## 1. 참조 논문

### 1.1 모델

- Gao et al., *An edge-deployable TinyML approach enhanced by transfer learning for efficient bearing fault diagnosis*, Sci China Tech Sci, 2025.
- 3축 가속도 진동 데이터(16 timestep × 6 채널)로 베어링 상태 4종(정상, 내륜, 볼, 외륜)을 분류하는 Res-CNN-LSTM.
   - 코드, 가중치, 자체 데이터셋은 공개되지 않음

### 1.2 배포 환경

| 배포 환경 | 내용 |
|---|---|
| 보드 | ESP32-S3-DevKitC-1 (Flash 16 MB) |
| 프레임워크 | Arduino IDE |
| 추론 런타임 | TFLite Micro |
| 양자화 | int8 PTQ |
| 경량화 | structural pruning (sparsity 30%) |
| CPU 클럭 | 160 MHz |
| 동시 실행 | Wi-Fi, MQTT, UDP 수신, 모델 2개 로드 (센서 원천에 따라 다른 모델 실행) |

### 1.3 보고 지표

| 지표 | 값 |
|---|---|
| 정확도 | 88.28% |
| 추론 시간 | 44.9 ms |
| RAM / Flash usage | 74.2 KB / 57.6 KB (측정 범위 미기재) |
| 모델 크기 | 73 KB (측정 기준 미기재) |

---

## 2. 재현 환경

### 2.1 측정 환경

| 측정 환경 | 내용 |
|---|---|
| 보드 | ESP32-S3-DevKitC-1 N16R8 (ESP32-S3 rev v0.2, Flash 16 MB, PSRAM 8 MB) |
| 프레임워크 | ESP-IDF v5.5.6 |
| 추론 런타임 | TFLite Micro (esp-tflite-micro 1.4.3) |
| 양자화 | int8 PTQ |
| 경량화 | 없음 |
| CPU 클럭 | 160 / 240 MHz |
| 실행 core | core 1 (Wi-Fi·esp_timer 태스크는 core 0) |
| 동시 실행 | 없음 (Wi-Fi off, 벤치마크만 실행) |

- 커널 : TFLM reference / ESP-NN 1.4.2 (PIE SIMD int8)
- 컴파일러 : GCC 14.2.0.
- 모델 변환 : TensorFlow 2.15.1, TFLite Converter

### 2.2 모델 재구성

- Supporting Information Table S8을 바탕으로 모델를 재구성, 파라미터 수(4,598) 일치
   - MaxPool의 stride는 Table S8의 출력 shape이 모두 (16, 32)로 유지되는 것을 근거로 1로 해석

| 레이어 | 출력 shape | 파라미터 | 설정 |
|---|---|---|---|
| Input | (16, 6) | 0 | |
| SeparableConv1D | (16, 32) | 242 | kernel=3, padding=same, activation=ReLU |
| MaxPooling1D | (16, 32) | 0 | pool=2, stride=1, padding=same |
| SeparableConv1D | (16, 32) | 1,152 | kernel=3, padding=same, activation=False |
| Add + ReLU | (16, 32) | 0 | |
| MaxPooling1D | (16, 32) | 0 | pool=2, stride=1, padding=same |
| LSTM | (16) | 3,136 | 16 unit, 16 step |
| Dense | (4) | 68 | softmax |
| **합계** | | **4,598** | |

- 가중치 : random
- 모델 크기 (`.tflite`) : int8 15.8 KB (16,136 B) / fp32 24.3 KB (24,928 B)

---

## 3. 실험

### 3.1 측정

- 초기화 후 워밍업 20회, 실측정 1000회 수행
- Latency: 입력 준비·출력 읽기 제외한 추론 시간(µs)과 CPU 사이클 측정
- Memory: tensor arena 실사용량, 정적 메모리(SRAM/Flash), 펌웨어 이미지 크기 측정
- op별 시간: profile build로 측정
   - `MicroProfiler`를 추가한 별도 빌드

### 3.2 조합

| 조합 | 커널 | CPU (MHz) |
|---|---|---|
| ref_160 | reference | 160 |
| ref_240 | reference | 240 |
| espnn_160 | ESP-NN | 160 |
| espnn_240 | ESP-NN | 240 |

- int8 모델
- ESP-IDF 기본 자원
   - Flash 2 MB (DIO 80 MHz) 
   - PSRAM off, ICache 16 KB / DCache 32 KB
   - 워치독 켬
- core 1에서 추론
- Wi-Fi off

---

## 4. 결과

### 4.1 추론 시간

| 조합 | 평균 (ms) | p99 (ms) | CPU 사이클 (×10⁶) |
|---|---|---|---|
| ref_160 | 25.04 | 25.07 | 4.01 |
| ref_240 | 16.95 | 16.97 | 4.07 |
| espnn_160 | 9.22 | 9.25 | 1.48 |
| espnn_240 | 6.61 | 6.62 | 1.59 |

- 커널 효과 (reference → ESP-NN): 2.6~2.7배
- CPU 클럭 효과 (160 → 240 MHz): 1.4~1.5배

### 4.2 op별 시간 (160 MHz, µs)

| op | reference | ESP-NN | 배수 |
|---|---|---|---|
| CONV_2D (pointwise 1×1) | 13,771 (55%) | 892 (10%) | 15.4 |
| UNIDIRECTIONAL_SEQUENCE_LSTM | 6,355 (25%) | 6,419 (70%) | 1.0 |
| DEPTHWISE_CONV_2D | 2,136 (9%) | 476 (5%) | 4.5 |
| MAX_POOL_2D | 1,742 (7%) | 141 (2%) | 12.4 |
| ADD | 680 (3%) | 268 (3%) | 2.5 |
| SOFTMAX | 139 (1%) | 693 (8%) | 0.20 |
| RESHAPE, STRIDED_SLICE, FULLY_CONNECTED | 257 | 339 | |
| **합계** | **25,080** | **9,228** | **2.7** |

- ESP-NN은 LSTM을 가속하지 않음 (ESP-NN 경로에서 LSTM이 70%를 차지)
- ESP-NN의 SOFTMAX는 reference보다 느림 (원인 미확인)

### 4.3 메모리와 바이너리 크기 (KB)

| 커널 | 모델 (Flash) | 펌웨어 이미지 (Flash) | arena 실사용 (RAM) | 정적 SRAM (RAM) | 남은 heap (RAM) |
|---|---|---|---|---|---|
| reference | 15.8 | 313.1 | 5.6 | 116.1 | 310.0 |
| ESP-NN | 15.8 | 355.6 | 7.2 | 116.2 | 309.7 |

- 정적 SRAM: 펌웨어 전체(앱, TFLM, ESP-IDF, FreeRTOS)가 빌드 시 확정하는 내부 SRAM. tensor arena 고정 예약 64 KB 포함.
- arena 실사용: 64 KB 예약 중 모델이 실제 사용한 양 (activation peak, scratch, 런타임 메타데이터).
- 모델 가중치는 Flash에서 직접 읽으므로 RAM을 사용하지 않음

---

## 5. 자원 설정 튜닝

가장 빠른 조합인 espnn_240을 기준으로 Flash 모드와 캐시 설정만 변경

- qio: Flash를 데이터 핀 4개로 읽는 모드 (Quad I/O, 클럭당 4비트). 기본값 DIO(2비트)보다 캐시 미스 한 번의 처리 시간이 약 절반
- ic32: ICache(Flash에 있는 코드를 담는 캐시)를 16 KB에서 32 KB로 확대.
- all: qio + ic32 + DCache(Flash에 있는 데이터를 담는 캐시) 64 KB, line 64 B.

### 5.1 추론 시간과 메모리

| 조합 | Flash | ICache | DCache / line | 평균 (ms) | 변화 | 남은 heap (KB) |
|---|---|---|---|---|---|---|
| espnn_240 (기준) | DIO | 16 KB | 32 KB / 32 B | 6.61 | | 309.7 |
| +qio | QIO | 16 KB | 32 KB / 32 B | 6.03 | -8.8% | 309.7 |
| +ic32 | DIO | 32 KB | 32 KB / 32 B | 5.38 | -18.6% | 293.7 |
| +qio_ic32 | QIO | 32 KB | 32 KB / 32 B | 5.31 | -19.5% | 293.7 |
| +all | QIO | 32 KB | 64 KB / 64 B | 5.31 | -19.6% | 262.4 |

- ICache 32 KB가 가장 효과적. 추론 중 실행되는 코드가 16 KB를 넘어 매 추론 반복되던 캐시 미스가 줄어듦
- ICache 32 KB에서는 QIO의 추가 효과가 작고, DCache 확대는 효과 없음
   - 가중치 15.8 KB가 이미 32 KB에 들어감
- 메모리 비용: ICache 32 KB는 IRAM 16 KB를 캐시로 전환해 heap -16 KB, DCache 64 KB는 heap -31 KB.

### 5.2 op별 시간 (µs)

| op | espnn_240 | +qio_ic32 | 변화 |
|---|---|---|---|
| UNIDIRECTIONAL_SEQUENCE_LSTM | 4,354 (66%) | 4,152 (77%) | -5% |
| CONV_2D | 684 | 456 | -33% |
| SOFTMAX | 526 | 363 | -31% |
| DEPTHWISE_CONV_2D | 405 | 180 | -56% |
| ADD | 220 | 99 | -55% |
| STRIDED_SLICE | 152 | 17 | -89% |
| MAX_POOL_2D | 120 | 71 | -41% |
| FULLY_CONNECTED | 109 | 19 | -83% |
| RESHAPE | 65 | 32 | -51% |
| **합계** | **6,635** | **5,389** | **-19%** |

- 개선의 84%는 LSTM 이외의 op에서 발생. 연산량이 작은 op일수록 기존 시간의 대부분이 캐시 미스
- LSTM은 연산이 주된 비용이라 거의 줄지 않음 (-5%). 튜닝 후 전체의 77%
- SOFTMAX는 31% 줄었지만 reference(240 MHz, 131 µs)보다 여전히 약 2.8배 느림. 캐시 미스 외에 커널 자체 비용도 큼

---

## 6. 논문과 비교

| 항목 | 논문 | 본 실험 |
|---|---|---|
| 추론 시간 | 44.9 ms | 6.61 ~ 25.04 ms |
| 모델 크기 (int8) | 73 KB (의미 불명) | 15.8 KB |
| RAM | 74.2 KB (의미 불명) | arena 5.6 ~ 7.2 KB |
| 프레임워크 | Arduino IDE | ESP-IDF v5.5.6 |
| 측정 시 Wi-Fi | 켜짐 | 꺼짐 |
