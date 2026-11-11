// int8 TFLM latency/memory benchmark (default model: Res-CNN-LSTM, Gao et al. 2025).
// Weights are random: latency and memory are determined by the graph (ops/shapes),
// not by weight values.
//
// Serial output (parsed by bench.py on the host):
//   CFG,<key>=<value>,...      build/runtime configuration
//   <MicroProfiler CSV>        per-op ticks for one inference (only with CONFIG_BENCH_PROFILE)
//   RESULT,<key>=<value>,...   latency statistics and memory usage
//   DONE
#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

#include "esp_chip_info.h"
#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_private/esp_clk.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

#include "model_data.h"

#if CONFIG_BENCH_WIFI
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/event_groups.h"
#include "nvs_flash.h"
#endif

#if defined(CONFIG_ESP_TFLITE_MICRO_USE_ESP_NN)
static const char* kKernel = "esp-nn";
#else
static const char* kKernel = "reference";
#endif

// System configuration, recorded in the CFG line so each result is self-describing.
#if CONFIG_COMPILER_OPTIMIZATION_PERF
static const char* kOpt = "O2";
#elif CONFIG_COMPILER_OPTIMIZATION_SIZE
static const char* kOpt = "Os";
#elif CONFIG_COMPILER_OPTIMIZATION_NONE
static const char* kOpt = "O0";
#else
static const char* kOpt = "Og";
#endif
// CONFIG_ESPTOOLPY_FLASHMODE is the image-header value ("dio" on S3 even when QIO is
// selected; the bootloader switches modes), so use the selection macros instead.
#if CONFIG_ESPTOOLPY_FLASHMODE_QIO
static const char* kFlashMode = "qio";
#elif CONFIG_ESPTOOLPY_FLASHMODE_QOUT
static const char* kFlashMode = "qout";
#elif CONFIG_ESPTOOLPY_FLASHMODE_DOUT
static const char* kFlashMode = "dout";
#else
static const char* kFlashMode = "dio";
#endif
#if CONFIG_SPIRAM
static const int kPsram = 1;
#else
static const int kPsram = 0;
#endif

#if CONFIG_BENCH_PROFILE
static const int kProfile = 1;
#else
static const int kProfile = 0;
#endif

namespace {
constexpr size_t kArenaSize = CONFIG_BENCH_ARENA_KB * 1024;
alignas(16) uint8_t g_arena[kArenaSize];  // internal SRAM (.bss)

#if CONFIG_BENCH_WIFI
EventGroupHandle_t s_wifi_events;
constexpr int kConnectedBit = BIT0;

void wifi_handler(void*, esp_event_base_t base, int32_t id, void*) {
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) esp_wifi_connect();
  if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) esp_wifi_connect();
  if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) xEventGroupSetBits(s_wifi_events, kConnectedBit);
}

bool wifi_start() {
  esp_err_t r = nvs_flash_init();
  if (r == ESP_ERR_NVS_NO_FREE_PAGES || r == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }
  s_wifi_events = xEventGroupCreate();
  esp_netif_init();
  esp_event_loop_create_default();
  esp_netif_create_default_wifi_sta();
  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  esp_wifi_init(&cfg);
  esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_handler, nullptr);
  esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_handler, nullptr);
  wifi_config_t wc = {};
  strncpy((char*)wc.sta.ssid, CONFIG_BENCH_WIFI_SSID, sizeof(wc.sta.ssid));
  strncpy((char*)wc.sta.password, CONFIG_BENCH_WIFI_PASSWORD, sizeof(wc.sta.password));
  esp_wifi_set_mode(WIFI_MODE_STA);
  esp_wifi_set_config(WIFI_IF_STA, &wc);
  esp_wifi_start();
  EventBits_t b = xEventGroupWaitBits(s_wifi_events, kConnectedBit, pdFALSE, pdTRUE, pdMS_TO_TICKS(20000));
  return b & kConnectedBit;
}
#endif

// Deterministic pseudo-random input (values do not affect latency)
void fill_input(TfLiteTensor* t, uint32_t seed) {
  int8_t* p = t->data.int8;
  for (size_t i = 0; i < t->bytes; ++i) {
    seed = seed * 1664525u + 1013904223u;
    p[i] = static_cast<int8_t>(seed >> 24);
  }
}

// Let the idle task run (outside any timed region) so the default task watchdog
// does not fire during long benchmark loops.
inline void yield_if_due(int i) {
  if (i % 10 == 9) vTaskDelay(1);
}
}  // namespace

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(1500));  // give the serial monitor time to attach
  bool wifi_ok = false;
#if CONFIG_BENCH_WIFI
  wifi_ok = wifi_start();
#endif

  const tflite::Model* model = tflite::GetModel(g_model_data);
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddReshape();
  resolver.AddDepthwiseConv2D();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAdd();
  resolver.AddUnidirectionalSequenceLSTM();
  resolver.AddStridedSlice();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

#if CONFIG_BENCH_PROFILE
  static tflite::MicroProfiler profiler;  // ~80 KB .bss, profile builds only
  static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize, nullptr, &profiler);
#define PROFILER_CLEAR() profiler.ClearEvents()
#else
  static tflite::MicroInterpreter interp(model, resolver, g_arena, kArenaSize);
#define PROFILER_CLEAR() ((void)0)
#endif
  if (interp.AllocateTensors() != kTfLiteOk) {
    printf("ERROR,AllocateTensors failed (arena %u KB too small?)\n", (unsigned)CONFIG_BENCH_ARENA_KB);
    return;
  }
  TfLiteTensor* in = interp.input(0);

  printf("CFG,opt=%s,flash_mode=%s,flash_freq=%s,psram=%d,icache_kb=%d,dcache_kb=%d,dcache_line=%d\n", kOpt,
         kFlashMode, CONFIG_ESPTOOLPY_FLASHFREQ, kPsram, CONFIG_ESP32S3_INSTRUCTION_CACHE_SIZE / 1024,
         CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024, CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE);
  printf("CFG,model=%s,kernel=%s,cpu_mhz=%d,core=%d,wifi=%s,wifi_connected=%d,model_bytes=%u,arena_kb=%d,runs=%d,profile=%d\n",
         BENCH_MODEL_NAME, kKernel, (int)(esp_clk_cpu_freq() / 1000000), (int)xPortGetCoreID(),
#if CONFIG_BENCH_WIFI
         "on",
#else
         "off",
#endif
         wifi_ok, g_model_data_len, CONFIG_BENCH_ARENA_KB, CONFIG_BENCH_RUNS, kProfile);

  // Warm-up (fill instruction/data caches)
  for (int i = 0; i < CONFIG_BENCH_WARMUP; ++i) {
    fill_input(in, i);
    interp.Invoke();
    yield_if_due(i);
  }

#if CONFIG_BENCH_PROFILE
  // Per-op profile of a single inference
  profiler.ClearEvents();
  fill_input(in, 12345);
  interp.Invoke();
  profiler.LogTicksPerTagCsv();
  profiler.ClearEvents();
#endif

  // Timed runs
  std::vector<uint32_t> us(CONFIG_BENCH_RUNS);
  uint64_t cycles_sum = 0;
  for (int i = 0; i < CONFIG_BENCH_RUNS; ++i) {
    fill_input(in, 1000 + i);
    uint32_t c0 = esp_cpu_get_cycle_count();
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus s = interp.Invoke();
    int64_t t1 = esp_timer_get_time();
    uint32_t c1 = esp_cpu_get_cycle_count();
    PROFILER_CLEAR();
    if (s != kTfLiteOk) {
      printf("ERROR,Invoke failed at %d\n", i);
      return;
    }
    us[i] = static_cast<uint32_t>(t1 - t0);
    cycles_sum += static_cast<uint32_t>(c1 - c0);
    yield_if_due(i);
  }
  std::vector<uint32_t> sorted = us;
  std::sort(sorted.begin(), sorted.end());
  double mean = 0;
  for (auto v : us) mean += v;
  mean /= us.size();
  auto pct = [&](double p) { return sorted[std::min(sorted.size() - 1, (size_t)(p * sorted.size()))]; };

  printf("RESULT,model=%s,kernel=%s,cpu_mhz=%d,wifi=%s,mean_us=%.1f,median_us=%" PRIu32 ",p99_us=%" PRIu32
         ",min_us=%" PRIu32 ",max_us=%" PRIu32 ",mean_cycles=%.0f,arena_used_bytes=%u,"
         "free_internal_heap=%u,min_free_internal_heap=%u\n",
         BENCH_MODEL_NAME, kKernel, (int)(esp_clk_cpu_freq() / 1000000),
#if CONFIG_BENCH_WIFI
         "on",
#else
         "off",
#endif
         mean, pct(0.5), pct(0.99), sorted.front(), sorted.back(), (double)cycles_sum / us.size(),
         (unsigned)interp.arena_used_bytes(),
         (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
         (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
  printf("DONE\n");
  fflush(stdout);
  while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}
