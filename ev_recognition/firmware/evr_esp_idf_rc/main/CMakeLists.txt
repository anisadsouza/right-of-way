idf_component_register(
    SRCS "main_rc.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_wifi esp_event esp_netif nvs_flash esp_http_server esp_timer
)