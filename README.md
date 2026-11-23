# mcu-ml-stack

ESP32-S3에서 TFLite Micro로 모델 추론 시간과 메모리를 측정하는 벤치마크.

## 구성

```
models/res_cnn_lstm/
  model.py              모델 구조 정의 (Keras)
  export.py             임의 가중치 → int8 .tflite → C 배열 (out/model_data.cc)
firmware/tflm_bench/    ESP-IDF 프로젝트 (보드에서 실행)
  main/main.cc          추론 측정 코드: 워밍업 20회 → 1000회 Invoke() 측정 → 시리얼 출력
  main/Kconfig.projbuild  앱 설정 항목 정의 (CONFIG_BENCH_*)
  sdkconfig.defaults    모든 조합 공통 설정 (-O2, 벤치마크 태스크 core 1)
  cfg/*.defaults        조합별 설정 조각 (커널, CPU 클럭, 캐시, Flash 모드 등)
  bench.py              조합별 빌드 → 플래시 → 결과 수집 (PC에서 실행)
docs/                   MCU·보드 설명, 실험 보고서
experiments/            SEU 데이터 전처리·학습 스크립트 (벤치마크에는 미사용)
```

## 요구 사항

| 항목 | 버전 / 사양 |
|---|---|
| 보드 | ESP32-S3-DevKitC-1 N16R8 (UART 포트로 PC 연결) |
| OS | Linux (Ubuntu 22.04에서 확인) |
| ESP-IDF | v5.5.6 (esp-tflite-micro 1.4.3, esp-nn 1.4.2는 빌드 시 자동 다운로드) |
| 모델 변환 | Python 3.11, TensorFlow 2.15.1 |

## 설치

### 1. 모델 변환용 Python 환경

```bash
conda create -n mcu python=3.11 -y
conda activate mcu
pip install tensorflow==2.15.1 numpy
```

TensorFlow 2.15(Keras 2)를 사용한다. Keras 3 기반 버전에서는 LSTM이 `UNIDIRECTIONAL_SEQUENCE_LSTM` 단일 연산으로 변환되지 않을 수 있다.

### 2. ESP-IDF

ESP-IDF 빌드에 필요한 시스템 패키지 (Ubuntu):

```bash
sudo apt-get install git wget flex bison gperf python3 python3-pip python3-venv \
  cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
```

```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.5.6 --recursive --depth 1 --shallow-submodules https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3
```

빌드·측정 전에 매번 환경을 불러온다.

```bash
. ~/esp/esp-idf/export.sh
```

### 3. 시리얼 포트 권한

플래시나 측정 중 `/dev/ttyUSB0` 접근이 거부(`Permission denied`)되면 사용자를 dialout 그룹에 추가한다.

```bash
sudo usermod -aG dialout $USER   # 이후 다시 로그인
```

보드의 UART 포트(CP2102)를 연결하면 `/dev/ttyUSB0`으로 잡힌다.
USB 포트로 연결한 경우 `/dev/ttyACM0` 등으로 잡히며, 이후 명령의 `-p` 값을 그에 맞게 바꾼다.

## 실행

### 1. 모델 변환

```bash
conda activate mcu
python models/res_cnn_lstm/export.py
```

`models/res_cnn_lstm/out/`에 `.tflite`와 `model_data.cc`가 생성된다. 펌웨어는 빌드 시 이 파일을 직접 사용한다.

### 2. 빌드

```bash
cd firmware/tflm_bench
. ~/esp/esp-idf/export.sh
python bench.py build              # 기본 4개 조합
python bench.py build --profile    # + op별 시간 측정용 *_prof 빌드
python bench.py build --tune       # + 자원 설정 튜닝 조합 (espnn_240_*)
```

### 3. 측정

```bash
python bench.py run -p /dev/ttyUSB0                          # 빌드된 조합 전부
python bench.py run -p /dev/ttyUSB0 --only ref_160 espnn_240 # 일부만
```

조합마다 플래시 → 리셋 → 시리얼 결과 수집을 자동으로 수행한다. `run`은 `build/`에 빌드되어 있는 조합을 모두 실행하므로, 일부만 돌릴 때는 `--only`로 지정한다.

### 4. 결과

```
results/res_cnn_lstm/bench_<timestamp>.csv      조합당 한 행 (추론 시간, CPU 사이클, arena 사용량, 정적 메모리 등)
results/res_cnn_lstm/<timestamp>_<조합>.log      조합별 시리얼 로그 (*_prof는 op별 시간 포함)
```

주요 CSV 열: `mean_us`, `median_us`, `p99_us`, `min_us`, `max_us`, `mean_cycles`, `arena_used_bytes`, `free_internal_heap`, `static_used_diram`, `static_total_size`.

## 실험 조합

| 옵션 | 조합 이름 | 내용 |
|---|---|---|
| (기본) | `ref_160`, `ref_240`, `espnn_160`, `espnn_240` | 커널(TFLM reference / ESP-NN) × CPU 클럭(160 / 240 MHz) |
| `--profile` | `*_prof` | 위 조합 + TFLM MicroProfiler (op별 시간) |
| `--tune` | `espnn_240_{qio,ic32,qio_ic32,all}` | Flash QIO, ICache 32 KB, DCache 64 KB / line 64 B 조합 |
| `--wifi` | `*_wifi` | Wi-Fi STA 연결 상태 |

공통: int8 모델, ESP-IDF 기본 자원 (Flash 2 MB DIO 80 MHz, PSRAM off, ICache 16 KB / DCache 32 KB), `-O2`, 벤치마크 태스크 core 1.

`--wifi`를 쓰려면 `firmware/tflm_bench/cfg/wifi_secrets.defaults`를 만든다 (git 제외, 2.4 GHz AP).

```
CONFIG_BENCH_WIFI_SSID="..."
CONFIG_BENCH_WIFI_PASSWORD="..."
```

## bench.py 없이 조합 하나 빌드·실행

```bash
cd firmware/tflm_bench
. ~/esp/esp-idf/export.sh
idf.py -B build/manual -DSDKCONFIG=build/manual/sdkconfig \
  -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;cfg/espnn.defaults;cfg/cpu240.defaults" \
  build flash monitor -p /dev/ttyUSB0
```

시리얼 출력의 `RESULT,` 줄에 측정 결과가 나온다. 설정 조각을 바꿀 때는 `build/manual/sdkconfig`를 지우고 다시 빌드한다 (기존 `sdkconfig`가 defaults보다 우선한다).
