/*
 * Emergency Vehicle Recognition - ESP32 4WD yield controller (native ESP-IDF).
 *
 * Commands arrive over UART0 (the USB cable to the Pi), one per line:
 * NORMAL, SLOW, PULL_RIGHT, STOP. Replies match the original Arduino sketch:
 * "OK <COMMAND>" or "Unknown command: <text>". The controller starts stopped.
 * Test with the wheels raised first.
 */

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include "esp_log.h"

#include "pins.h"

#define CMD_UART      UART_NUM_0
#define CMD_BAUD      115200
#define RX_BUF_SIZE   256
#define LINE_MAX_LEN  64

#define LEFT_CHANNEL   LEDC_CHANNEL_0
#define RIGHT_CHANNEL  LEDC_CHANNEL_1
#define PWM_MODE       LEDC_LOW_SPEED_MODE

static void reply(const char *text)
{
    uart_write_bytes(CMD_UART, text, strlen(text));
    uart_write_bytes(CMD_UART, "\r\n", 2);
}

static void init_outputs(void)
{
    const gpio_num_t outputs[] = {
        PIN_GREEN_LED, PIN_YELLOW_LED, PIN_RED_LED, PIN_BUZZER,
        PIN_LEFT_IN1, PIN_LEFT_IN2, PIN_RIGHT_IN3, PIN_RIGHT_IN4,
    };
    for (size_t i = 0; i < sizeof(outputs) / sizeof(outputs[0]); i++) {
        gpio_reset_pin(outputs[i]);
        gpio_set_direction(outputs[i], GPIO_MODE_OUTPUT);
        gpio_set_level(outputs[i], 0);
    }

    ledc_timer_config_t timer = {
        .speed_mode = PWM_MODE,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = PWM_FREQUENCY_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&timer);

    ledc_channel_config_t left = {
        .gpio_num = PIN_LEFT_ENABLE,
        .speed_mode = PWM_MODE,
        .channel = LEFT_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0,
    };
    ledc_channel_config(&left);

    ledc_channel_config_t right = left;
    right.gpio_num = PIN_RIGHT_ENABLE;
    right.channel = RIGHT_CHANNEL;
    ledc_channel_config(&right);
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
    ledc_set_duty(PWM_MODE, channel, duty);
    ledc_update_duty(PWM_MODE, channel);
}

static void drive(int left_speed, int right_speed)
{
    gpio_set_level(PIN_LEFT_IN1, left_speed >= 0);
    gpio_set_level(PIN_LEFT_IN2, left_speed < 0);
    gpio_set_level(PIN_RIGHT_IN3, right_speed >= 0);
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

static void handle_command(const char *command)
{
    char response[LINE_MAX_LEN + 32];

    if (strcmp(command, "NORMAL") == 0) {
        set_lights(true, false, false);
        gpio_set_level(PIN_BUZZER, 0);
        drive(160, 160);
    } else if (strcmp(command, "SLOW") == 0) {
        set_lights(false, true, false);
        gpio_set_level(PIN_BUZZER, 1);
        drive(90, 90);
    } else if (strcmp(command, "PULL_RIGHT") == 0) {
        /* Left side faster than right, producing a right arc. */
        set_lights(false, true, true);
        gpio_set_level(PIN_BUZZER, 1);
        drive(110, 35);
    } else if (strcmp(command, "STOP") == 0) {
        set_lights(false, false, true);
        gpio_set_level(PIN_BUZZER, 1);
        stop_motors();
    } else {
        snprintf(response, sizeof(response), "Unknown command: %s", command);
        reply(response);
        return;
    }
    snprintf(response, sizeof(response), "OK %s", command);
    reply(response);
}

static void init_uart(void)
{
    uart_config_t config = {
        .baud_rate = CMD_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_driver_install(CMD_UART, RX_BUF_SIZE, 0, 0, NULL, 0);
    uart_param_config(CMD_UART, &config);
}

void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_NONE);
    init_outputs();
    init_uart();
    handle_command("STOP");
    reply("EVR ESP32 4WD controller ready");

    char line[LINE_MAX_LEN];
    size_t length = 0;
    uint8_t byte;

    while (true) {
        if (uart_read_bytes(CMD_UART, &byte, 1, portMAX_DELAY) != 1) {
            continue;
        }
        if (byte == '\n') {
            line[length] = '\0';
            /* Trim trailing carriage return and spaces, then upper-case. */
            while (length > 0 && isspace((unsigned char)line[length - 1])) {
                line[--length] = '\0';
            }
            size_t start = 0;
            while (line[start] != '\0' && isspace((unsigned char)line[start])) {
                start++;
            }
            for (size_t i = start; line[i] != '\0'; i++) {
                line[i] = (char)toupper((unsigned char)line[i]);
            }
            if (line[start] != '\0') {
                handle_command(&line[start]);
            }
            length = 0;
        } else if (length < LINE_MAX_LEN - 1) {
            line[length++] = (char)byte;
        }
    }
}
