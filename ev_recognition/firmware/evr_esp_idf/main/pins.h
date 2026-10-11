#pragma once

#include "driver/gpio.h"

/* Traffic-light module and buzzer */
#define PIN_GREEN_LED   GPIO_NUM_32
#define PIN_YELLOW_LED  GPIO_NUM_33
#define PIN_RED_LED     GPIO_NUM_23
#define PIN_BUZZER      GPIO_NUM_4

/* L298N left pair: OUT1/OUT2. Remove the ENA jumper for PWM control. */
#define PIN_LEFT_ENABLE   GPIO_NUM_25
#define PIN_LEFT_IN1      GPIO_NUM_26
#define PIN_LEFT_IN2      GPIO_NUM_27

/* L298N right pair: OUT3/OUT4. Remove the ENB jumper for PWM control. */
#define PIN_RIGHT_ENABLE  GPIO_NUM_14
#define PIN_RIGHT_IN3     GPIO_NUM_18
#define PIN_RIGHT_IN4     GPIO_NUM_19

#define PWM_FREQUENCY_HZ  1000
