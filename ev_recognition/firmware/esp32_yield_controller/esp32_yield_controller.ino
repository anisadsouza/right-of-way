/*
  Emergency Vehicle Recognition - ESP32 yield controller

  Serial commands from Raspberry Pi:
    NORMAL     green LED, normal drive
    SLOW       yellow LED, reduced motor speed
    PULL_RIGHT yellow + red LED, steer/pull right
    STOP       red LED, motors stopped
*/

#include <Arduino.h>
#include <ESP32Servo.h>

const int GREEN_LED = 25;
const int YELLOW_LED = 26;
const int RED_LED = 27;
const int BUZZER = 14;

const int LEFT_PWM = 18;
const int LEFT_IN1 = 19;
const int LEFT_IN2 = 21;
const int RIGHT_PWM = 5;
const int RIGHT_IN1 = 17;
const int RIGHT_IN2 = 16;
const int SERVO_PIN = 13;

const int PWM_FREQ = 1000;
const int PWM_RESOLUTION = 8;
const int LEFT_CH = 0;
const int RIGHT_CH = 1;

Servo steering;

void setDrive(int leftSpeed, int rightSpeed);
void setLights(bool green, bool yellow, bool red);
void handleCommand(String command);

void setup() {
  Serial.begin(115200);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);

  ledcSetup(LEFT_CH, PWM_FREQ, PWM_RESOLUTION);
  ledcSetup(RIGHT_CH, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(LEFT_PWM, LEFT_CH);
  ledcAttachPin(RIGHT_PWM, RIGHT_CH);

  steering.attach(SERVO_PIN);
  handleCommand("NORMAL");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    command.toUpperCase();
    handleCommand(command);
  }
}

void handleCommand(String command) {
  if (command == "NORMAL") {
    setLights(true, false, false);
    noTone(BUZZER);
    steering.write(90);
    setDrive(160, 160);
  } else if (command == "SLOW") {
    setLights(false, true, false);
    tone(BUZZER, 1200, 120);
    steering.write(90);
    setDrive(85, 85);
  } else if (command == "PULL_RIGHT") {
    setLights(false, true, true);
    tone(BUZZER, 1500, 150);
    steering.write(125);
    setDrive(75, 45);
  } else if (command == "STOP") {
    setLights(false, false, true);
    tone(BUZZER, 900, 250);
    steering.write(90);
    setDrive(0, 0);
  }
}

void setLights(bool green, bool yellow, bool red) {
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(YELLOW_LED, yellow ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
}

void setDrive(int leftSpeed, int rightSpeed) {
  digitalWrite(LEFT_IN1, leftSpeed > 0 ? HIGH : LOW);
  digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN1, rightSpeed > 0 ? HIGH : LOW);
  digitalWrite(RIGHT_IN2, LOW);
  ledcWrite(LEFT_CH, constrain(abs(leftSpeed), 0, 255));
  ledcWrite(RIGHT_CH, constrain(abs(rightSpeed), 0, 255));
}
