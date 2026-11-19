
/*
 * Project 27 - RC + AUTO controller
 * ESP32-WROVER / ESP-IDF 5.5.x
 *
 * AUTO:
 *   Raspberry Pi commands over UART0.
 *
 * MANUAL:
 *   Phone commands over the ESP32 Wi-Fi access point.
 *
 * The original main.c is not used by this RC build.
 * Original project remains untouched.
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/uart.h"

#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/timers.h"

#include "pins.h"

/* ---------------- CONFIGURATION ---------------- */

#define CMD_UART       UART_NUM_0
#define CMD_BAUD       115200
#define RX_BUF_SIZE    256
#define LINE_MAX_LEN   64

#define LEFT_CHANNEL   LEDC_CHANNEL_0
#define RIGHT_CHANNEL  LEDC_CHANNEL_1
#define PWM_MODE       LEDC_LOW_SPEED_MODE

#define BUZZER_DURATION_MS 2000
#define MANUAL_TIMEOUT_US  700000

#define WIFI_SSID      "PROJECT27_RC"
#define WIFI_PASSWORD  "P27Car2026"

/* ---------------- CONTROL STATE ---------------- */

typedef enum {
    MODE_AUTO = 0,
    MODE_MANUAL
} control_mode_t;

static control_mode_t current_mode = MODE_AUTO;

static SemaphoreHandle_t control_mutex = NULL;
static TimerHandle_t buzzer_timer = NULL;
static httpd_handle_t web_server = NULL;

static bool manual_moving = false;
static int64_t last_manual_command_us = 0;

/*
 * Remember the latest Pi command.
 * This allows AUTO mode to resume correctly after MANUAL mode.
 */
static char last_pi_command[LINE_MAX_LEN] = "STOP";

/* ---------------- MOTOR AND OUTPUT CONTROL ---------------- */

static void buzzer_timeout(TimerHandle_t timer)
{
    (void)timer;
    gpio_set_level(PIN_BUZZER, 1);
}

static void buzzer_on_for_a_few_seconds(void)
{
    xTimerStop(buzzer_timer, 0);
    gpio_set_level(PIN_BUZZER, 0);
    xTimerStart(buzzer_timer, 0);
}

static void buzzer_off(void)
{
    xTimerStop(buzzer_timer, 0);
    gpio_set_level(PIN_BUZZER, 1);
}

static void init_outputs(void)
{
    const gpio_num_t outputs[] = {
        PIN_GREEN_LED,
        PIN_YELLOW_LED,
        PIN_RED_LED,
        PIN_BUZZER,
        PIN_LEFT_IN1,
        PIN_LEFT_IN2,
        PIN_RIGHT_IN3,
        PIN_RIGHT_IN4
    };

    for (size_t i = 0; i < sizeof(outputs) / sizeof(outputs[0]); i++) {
        ESP_ERROR_CHECK(gpio_reset_pin(outputs[i]));
        ESP_ERROR_CHECK(gpio_set_direction(outputs[i], GPIO_MODE_OUTPUT));

        ESP_ERROR_CHECK(gpio_set_level(
            outputs[i],
            outputs[i] == PIN_BUZZER ? 1 : 0
        ));
    }

    ledc_timer_config_t timer = {
        .speed_mode = PWM_MODE,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = PWM_FREQUENCY_HZ,
        .clk_cfg = LEDC_AUTO_CLK
    };

    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    ledc_channel_config_t left = {
        .gpio_num = PIN_LEFT_ENABLE,
        .speed_mode = PWM_MODE,
        .channel = LEFT_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0
    };

    ESP_ERROR_CHECK(ledc_channel_config(&left));

    ledc_channel_config_t right = left;
    right.gpio_num = PIN_RIGHT_ENABLE;
    right.channel = RIGHT_CHANNEL;

    ESP_ERROR_CHECK(ledc_channel_config(&right));
}

static void set_lights(bool green, bool yellow, bool red)
{
    gpio_set_level(PIN_GREEN_LED, green);
    gpio_set_level(PIN_YELLOW_LED, yellow);
    gpio_set_level(PIN_RED_LED, red);
}

static void set_pwm(ledc_channel_t channel, int speed)
{
    int duty = abs(speed);

    if (duty > 255) {
        duty = 255;
    }

    ESP_ERROR_CHECK(ledc_set_duty(PWM_MODE, channel, duty));
    ESP_ERROR_CHECK(ledc_update_duty(PWM_MODE, channel));
}

static void drive(int left_speed, int right_speed)
{
    gpio_set_level(PIN_LEFT_IN1, left_speed > 0);
    gpio_set_level(PIN_LEFT_IN2, left_speed < 0);

    gpio_set_level(PIN_RIGHT_IN3, right_speed > 0);
    gpio_set_level(PIN_RIGHT_IN4, right_speed < 0);

    set_pwm(LEFT_CHANNEL, left_speed);
    set_pwm(RIGHT_CHANNEL, right_speed);
}

static void stop_motors(void)
{
    gpio_set_level(PIN_LEFT_IN1, 0);
    gpio_set_level(PIN_LEFT_IN2, 0);

    gpio_set_level(PIN_RIGHT_IN3, 0);
    gpio_set_level(PIN_RIGHT_IN4, 0);

    set_pwm(LEFT_CHANNEL, 0);
    set_pwm(RIGHT_CHANNEL, 0);
}

/* ---------------- AUTO MODE ---------------- */

/*
 * This function is called while control_mutex is already held.
 * It applies the remembered Pi command without taking the mutex again.
 */
static void apply_auto_command_locked(const char *command)
{
    if (strcmp(command, "NORMAL") == 0) {
        set_lights(true, false, false);
        buzzer_off();
        drive(160, 160);
    }
    else if (strcmp(command, "SLOW") == 0) {
        set_lights(false, true, false);
        buzzer_off();
        drive(90, 90);
    }
    else if (strcmp(command, "PULL_RIGHT") == 0) {
        set_lights(false, true, true);
        buzzer_on_for_a_few_seconds();
        drive(110, 35);
    }
    else {
        /* STOP or unknown remembered state: remain stopped. */
        stop_motors();
        buzzer_off();
        set_lights(false, false, true);
    }
}

static void apply_pi_command(const char *command)
{
    xSemaphoreTake(control_mutex, portMAX_DELAY);

    strncpy(last_pi_command, command, sizeof(last_pi_command) - 1);
    last_pi_command[sizeof(last_pi_command) - 1] = '\0';

    /*
     * STOP is always accepted.
     * Other Pi commands are applied only in AUTO mode.
     */
    if (strcmp(command, "STOP") == 0) {
        manual_moving = false;
        stop_motors();
        set_lights(false, false, true);
        buzzer_on_for_a_few_seconds();
    }
    else if (current_mode == MODE_AUTO) {
        apply_auto_command_locked(command);
    }

    xSemaphoreGive(control_mutex);
}

/* ---------------- MANUAL MODE: PHONE ---------------- */

static void apply_manual_command(const char *command)
{
    xSemaphoreTake(control_mutex, portMAX_DELAY);

    /*
     * STOP is accepted regardless of the current mode.
     */
    if (strcmp(command, "STOP") == 0) {
        manual_moving = false;
        stop_motors();
        buzzer_off();

        if (current_mode == MODE_MANUAL) {
            set_lights(false, false, false);
        }
    }
    else if (current_mode == MODE_MANUAL) {
        buzzer_off();
        set_lights(false, false, false);

        if (strcmp(command, "FORWARD") == 0) {
            drive(160, 160);
        }
        else if (strcmp(command, "BACKWARD") == 0) {
            drive(-160, -160);
        }
        else if (strcmp(command, "LEFT") == 0) {
            drive(-140, 140);
        }
        else if (strcmp(command, "RIGHT") == 0) {
            drive(140, -140);
        }
        else {
            xSemaphoreGive(control_mutex);
            return;
        }

        manual_moving = true;
        last_manual_command_us = esp_timer_get_time();
    }

    xSemaphoreGive(control_mutex);
}

/* ---------------- MODE MANAGEMENT ---------------- */

static void change_mode(control_mode_t new_mode)
{
    xSemaphoreTake(control_mutex, portMAX_DELAY);

    /* Always stop before changing operating modes. */
    stop_motors();
    buzzer_off();
    manual_moving = false;

    current_mode = new_mode;

    if (new_mode == MODE_AUTO) {
    /*
     * Safety: never resume movement using a stale Pi command.
     * Wait for a fresh command from the Raspberry Pi.
     */
    stop_motors();
    set_lights(false, false, true);

    }
    else {
        set_lights(false, false, false);
    }

    xSemaphoreGive(control_mutex);
}

/* ---------------- PHONE WEBPAGE ---------------- */

static const char index_html[] =
"<!DOCTYPE html>"
"<html><head>"
"<meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>Project 27 RC</title>"
"<style>"
"body{font-family:Arial;background:#101827;color:white;text-align:center;"
"margin:0;padding:22px}"
"h2{margin-bottom:8px}"
".sub{color:#b7c5d8;margin-bottom:22px}"
".modes{display:flex;gap:12px;justify-content:center;margin-bottom:24px}"
"button{border:0;border-radius:15px;font-size:19px;font-weight:bold;"
"color:white;padding:18px;touch-action:none;user-select:none}"
".mode{background:#35445c}"
".active{background:#16845c}"
".pad{display:grid;grid-template-columns:repeat(3,85px);gap:12px;"
"justify-content:center}"
".move{background:#246fc1;height:75px}"
".stop{background:#c53939;height:75px}"
".empty{visibility:hidden}"
".hint{color:#b7c5d8;font-size:13px;margin-top:24px}"
"</style></head><body>"

"<h2>PROJECT 27</h2>"
"<div class='sub'>Mobile Remote Controller</div>"

"<div class='modes'>"
"<button id='auto' class='mode active' onclick=\"setMode('AUTO')\">AUTO</button>"
"<button id='manual' class='mode' onclick=\"setMode('MANUAL')\">MANUAL</button>"
"</div>"

"<div id='status'>Mode: AUTO</div><br>"

"<div class='pad'>"
"<button class='empty'> </button>"
"<button class='move' data-move='FORWARD'>UP</button>"
"<button class='empty'> </button>"

"<button class='move' data-move='LEFT'>LEFT</button>"
"<button class='stop' onclick='sendStop()'>STOP</button>"
"<button class='move' data-move='RIGHT'>RIGHT</button>"

"<button class='empty'> </button>"
"<button class='move' data-move='BACKWARD'>DOWN</button>"
"<button class='empty'> </button>"
"</div>"

"<div class='hint'>Hold a direction to move. Release to stop.</div>"

"<script>"
"let repeat=null;"

"function sendMove(m){"
"fetch('/cmd?move='+m,{cache:'no-store'}).catch(()=>{});"
"}"

"function sendStop(){"
"if(repeat){clearInterval(repeat);repeat=null;}"
"sendMove('STOP');"
"}"

"async function setMode(m){"
"if(repeat){clearInterval(repeat);repeat=null;}"
"try{"
"await fetch('/cmd?move=STOP',{cache:'no-store'});"
"await fetch('/mode?set='+m,{cache:'no-store'});"
"document.getElementById('status').textContent='Mode: '+m;"
"document.getElementById('auto').className="
"m==='AUTO'?'mode active':'mode';"
"document.getElementById('manual').className="
"m==='MANUAL'?'mode active':'mode';"
"}catch(e){"
"document.getElementById('status').textContent='Connection error';"
"}"
"}"

"document.querySelectorAll('[data-move]').forEach(b=>{"
"b.addEventListener('pointerdown',e=>{"
"e.preventDefault();"
"sendMove(b.dataset.move);"
"if(repeat)clearInterval(repeat);"
"repeat=setInterval(()=>sendMove(b.dataset.move),150);"
"});"

"['pointerup','pointercancel','pointerleave'].forEach(ev=>{"
"b.addEventListener(ev,e=>{e.preventDefault();sendStop();});"
"});"
"});"

"window.addEventListener('blur',sendStop);"
"</script></body></html>";

/* ---------------- HTTP REQUEST HANDLERS ---------------- */

static esp_err_t root_handler(httpd_req_t *request)
{
    httpd_resp_set_type(request, "text/html");
    httpd_resp_set_hdr(request, "Cache-Control", "no-store");

    return httpd_resp_send(
        request,
        index_html,
        HTTPD_RESP_USE_STRLEN
    );
}

static esp_err_t command_handler(httpd_req_t *request)
{
    char query[64];
    char move[24];

    if (httpd_req_get_url_query_str(
            request, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(
            query, "move", move, sizeof(move)) != ESP_OK) {

        httpd_resp_send_err(
            request,
            HTTPD_400_BAD_REQUEST,
            "Missing move"
        );

        return ESP_OK;
    }

    for (size_t i = 0; move[i]; i++) {
        move[i] = (char)toupper((unsigned char)move[i]);
    }

    if (strcmp(move, "FORWARD") == 0 ||
        strcmp(move, "BACKWARD") == 0 ||
        strcmp(move, "LEFT") == 0 ||
        strcmp(move, "RIGHT") == 0 ||
        strcmp(move, "STOP") == 0) {

        apply_manual_command(move);
        httpd_resp_sendstr(request, "OK");
    }
    else {
        httpd_resp_send_err(
            request,
            HTTPD_400_BAD_REQUEST,
            "Unknown command"
        );
    }

    return ESP_OK;
}

static esp_err_t mode_handler(httpd_req_t *request)
{
    char query[64];
    char mode[24];

    if (httpd_req_get_url_query_str(
            request, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(
            query, "set", mode, sizeof(mode)) != ESP_OK) {

        httpd_resp_send_err(
            request,
            HTTPD_400_BAD_REQUEST,
            "Missing mode"
        );

        return ESP_OK;
    }

    for (size_t i = 0; mode[i]; i++) {
        mode[i] = (char)toupper((unsigned char)mode[i]);
    }

    if (strcmp(mode, "AUTO") == 0) {
        change_mode(MODE_AUTO);
    }
    else if (strcmp(mode, "MANUAL") == 0) {
        change_mode(MODE_MANUAL);
    }
    else {
        httpd_resp_send_err(
            request,
            HTTPD_400_BAD_REQUEST,
            "Unknown mode"
        );

        return ESP_OK;
    }

    httpd_resp_sendstr(request, "OK");
    return ESP_OK;
}

/* ---------------- WEB SERVER ---------------- */

static void start_web_server(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.stack_size = 8192;

    ESP_ERROR_CHECK(httpd_start(&web_server, &config));

    httpd_uri_t root = {
        .uri = "/",
        .method = HTTP_GET,
        .handler = root_handler
    };

    httpd_uri_t command = {
        .uri = "/cmd",
        .method = HTTP_GET,
        .handler = command_handler
    };

    httpd_uri_t mode = {
        .uri = "/mode",
        .method = HTTP_GET,
        .handler = mode_handler
    };

    ESP_ERROR_CHECK(httpd_register_uri_handler(web_server, &root));
    ESP_ERROR_CHECK(httpd_register_uri_handler(web_server, &command));
    ESP_ERROR_CHECK(httpd_register_uri_handler(web_server, &mode));
}

/* ---------------- WI-FI ACCESS POINT ---------------- */

static void start_wifi(void)
{
    esp_err_t ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {

        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
    else {
        ESP_ERROR_CHECK(ret);
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(
        (esp_netif_create_default_wifi_ap() != NULL)
            ? ESP_OK : ESP_FAIL
    );

    wifi_init_config_t wifi_init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&wifi_init));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid = WIFI_SSID,
            .ssid_len = sizeof(WIFI_SSID) - 1,
            .channel = 1,
            .password = WIFI_PASSWORD,
            .max_connection = 2,
            .authmode = WIFI_AUTH_WPA2_PSK
        }
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    start_web_server();
}

/* ---------------- MANUAL COMMAND WATCHDOG ---------------- */

static void watchdog_task(void *argument)
{
    (void)argument;

    while (true) {
        vTaskDelay(pdMS_TO_TICKS(100));

        xSemaphoreTake(control_mutex, portMAX_DELAY);

        if (current_mode == MODE_MANUAL && manual_moving) {
            int64_t now = esp_timer_get_time();

            if (now - last_manual_command_us > MANUAL_TIMEOUT_US) {
                stop_motors();
                manual_moving = false;
            }
        }

        xSemaphoreGive(control_mutex);
    }
}

/* ---------------- UART COMMUNICATION WITH PI ---------------- */

static void reply(const char *text)
{
    uart_write_bytes(CMD_UART, text, strlen(text));
    uart_write_bytes(CMD_UART, "\r\n", 2);
}

static void init_uart(void)
{
    uart_config_t config = {
        .baud_rate = CMD_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT
    };

    ESP_ERROR_CHECK(
        uart_driver_install(
            CMD_UART,
            RX_BUF_SIZE,
            0,
            0,
            NULL,
            0
        )
    );

    ESP_ERROR_CHECK(uart_param_config(CMD_UART, &config));
}

static void process_uart_command(const char *command)
{
    if (strcmp(command, "NORMAL") == 0 ||
        strcmp(command, "SLOW") == 0 ||
        strcmp(command, "PULL_RIGHT") == 0 ||
        strcmp(command, "STOP") == 0) {

        apply_pi_command(command);

        /*
         * The Pi controller does not wait for an acknowledgement,
         * but retain a simple OK response for compatibility.
         */
        reply("OK");
    }
    else {
        char response[LINE_MAX_LEN + 32];

        snprintf(
            response,
            sizeof(response),
            "Unknown command: %s",
            command
        );

        reply(response);
    }
}

/* ---------------- APPLICATION ENTRY POINT ---------------- */

void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_NONE);

    control_mutex = xSemaphoreCreateMutex();

    if (control_mutex == NULL) {
        return;
    }

    init_outputs();
    init_uart();

    buzzer_timer = xTimerCreate(
        "buzzer_timer",
        pdMS_TO_TICKS(BUZZER_DURATION_MS),
        pdFALSE,
        NULL,
        buzzer_timeout
    );

    if (buzzer_timer == NULL) {
        return;
    }

    /* Start stopped. AUTO is the default operating mode. */
    stop_motors();
    set_lights(false, false, true);

    start_wifi();

    BaseType_t task_result = xTaskCreate(
        watchdog_task,
        "manual_watchdog",
        3072,
        NULL,
        5,
        NULL
    );

    if (task_result != pdPASS) {
        stop_motors();
        return;
    }

    reply("EVR ESP32 RC controller ready");

    char line[LINE_MAX_LEN];
    size_t length = 0;
    uint8_t byte;

    while (true) {
        if (uart_read_bytes(
                CMD_UART,
                &byte,
                1,
                portMAX_DELAY
            ) != 1) {
            continue;
        }

        if (byte == '\n') {
            line[length] = '\0';

            while (length > 0 &&
                   isspace((unsigned char)line[length - 1])) {
                line[--length] = '\0';
            }

            size_t start = 0;

            while (line[start] != '\0' &&
                   isspace((unsigned char)line[start])) {
                start++;
            }

            for (size_t i = start; line[i] != '\0'; i++) {
                line[i] = (char)toupper((unsigned char)line[i]);
            }

            if (line[start] != '\0') {
                process_uart_command(&line[start]);
            }

            length = 0;
        }
        else if (length < LINE_MAX_LEN - 1) {
            line[length++] = (char)byte;
        }
        else {
            /* Discard an overlong command line. */
            length = 0;
        }
    }
}
