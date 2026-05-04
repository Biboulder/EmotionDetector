idf_component_register(
    SRCS "main.cpp" "camera.cpp" "inference.cpp" "preprocess.cpp" "model.c"
    INCLUDE_DIRS "."
    REQUIRES nvs_flash esp_timer esp_system driver log
)
