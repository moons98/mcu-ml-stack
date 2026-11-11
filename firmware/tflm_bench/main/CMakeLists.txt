# Model under test: models/<BENCH_MODEL>/out/model_data.{cc,h}
#   idf.py -DBENCH_MODEL=res_cnn_lstm build   (default: res_cnn_lstm)
if(NOT DEFINED BENCH_MODEL)
    set(BENCH_MODEL res_cnn_lstm)
endif()
get_filename_component(MODEL_OUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../models/${BENCH_MODEL}/out" ABSOLUTE)
if(NOT EXISTS "${MODEL_OUT_DIR}/model_data.cc")
    message(FATAL_ERROR "${MODEL_OUT_DIR}/model_data.cc not found; run models/${BENCH_MODEL}/export.py first")
endif()

idf_component_register(
    SRCS "main.cc" "${MODEL_OUT_DIR}/model_data.cc"
    INCLUDE_DIRS "." "${MODEL_OUT_DIR}"
    PRIV_REQUIRES esp_timer esp_wifi nvs_flash esp_netif esp_event esp_hw_support
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE BENCH_MODEL_NAME="${BENCH_MODEL}")
