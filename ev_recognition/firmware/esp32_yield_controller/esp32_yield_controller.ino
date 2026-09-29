/*
  Emergency Vehicle Recognition - ESP32 4WD yield controller

  Hardware: ESP32-DevKitC, L298N, four DC motors wired as left pair/right pair,
  active buzzer, and 5 V traffic-light module. The Pi sends commands through
  the ESP32 USB cable, one command per line: NORMAL, SLOW, PULL_RIGHT, STOP.

  The controller starts stopped. Test it with the wheels raised first.
*/

#include <Arduino.h>

const int GREEN_LED = 32;
const int YELLOW_LED = 33;
const int RED_LED = 23;
const int BUZZER = 4;

// L298N: OUT1/OUT2 -> both left motors, OUT3/OUT4 -> both right motors.
const int LEFT_ENABLE = 25;  // Remove L298N ENA jumper for PWM control.
const int LEFT_IN1 = 26;
const int LEFT_IN2 = 27;
const int RIGHT_ENABLE = 14; // Remove L298N ENB jumper for PWM control.
const int RIGHT_IN3 = 18;
const int RIGHT_IN4 = 19;

const int PWM_FREQUENCY = 1000;
const int PWM_RESOLUTION = 8;
const int LEFT_PWM_CHANNEL = 0;
const int RIGHT_PWM_CHANNEL = 1;

void setLights(bool green, bool yellow, bool red);
void drive(int leftSpeed, int rightSpeed);
void stopMotors();
void handleCommand(String command);

void setup() {
  Serial.begin(115200);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(RIGHT_IN3, OUTPUT);
  pinMode(RIGHT_IN4, OUTPUT);
  ledcSetup(LEFT_PWM_CHANNEL, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcSetup(RIGHT_PWM_CHANNEL, PWM_FREQUENCY, PWM_RESOLUTION);
  ledcAttachPin(LEFT_ENABLE, LEFT_PWM_CHANNEL);
  ledcAttachPin(RIGHT_ENABLE, RIGHT_PWM_CHANNEL);
  handleCommand("STOP");
  Serial.println("EVR ESP32 4WD controller ready");
}

void loop() {
  if (!Serial.available()) return;
  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();
  handleCommand(command);
}

void handleCommand(String command) {
  if (command == "NORMAL") {
    setLights(true, false, false);
    digitalWrite(BUZZER, LOW);
    drive(160, 160);
  } else if (command == "SLOW") {
    setLights(false, true, false);
    digitalWrite(BUZZER, HIGH);
    drive(90, 90);
  } else if (command == "PULL_RIGHT") {
    // Differential drive: left side moves faster, producing a right arc.
    setLights(false, true, true);
    digitalWrite(BUZZER, HIGH);
    drive(110, 35);
  } else if (command == "STOP") {
    setLights(false, false, true);
    digitalWrite(BUZZER, HIGH);
    stopMotors();
  } else {
    Serial.print("Unknown command: ");
    Serial.println(command);
    return;
  }
  Serial.print("OK ");
  Serial.println(command);
}

void setLights(bool green, bool yellow, bool red) {
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(YELLOW_LED, yellow ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
}

void drive(int leftSpeed, int rightSpeed) {
  digitalWrite(LEFT_IN1, leftSpeed >= 0 ? HIGH : LOW);
  digitalWrite(LEFT_IN2, leftSpeed >= 0 ? LOW : HIGH);
  digitalWrite(RIGHT_IN3, rightSpeed >= 0 ? HIGH : LOW);
  digitalWrite(RIGHT_IN4, rightSpeed >= 0 ? LOW : HIGH);
  ledcWrite(LEFT_PWM_CHANNEL, constrain(abs(leftSpeed), 0, 255));
  ledcWrite(RIGHT_PWM_CHANNEL, constrain(abs(rightSpeed), 0, 255));
}

void stopMotors() {
  digitalWrite(LEFT_IN1, LOW);
  digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN3, LOW);
  digitalWrite(RIGHT_IN4, LOW);
  ledcWrite(LEFT_PWM_CHANNEL, 0);
  ledcWrite(RIGHT_PWM_CHANNEL, 0);
}
