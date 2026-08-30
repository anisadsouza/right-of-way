/*
  Emergency Vehicle Recognition - ESP32 traffic light controller

  Raspberry Pi sends one command per line over USB serial:
    NORMAL      green on
    SLOW        yellow on
    PULL_RIGHT  yellow + red on
    STOP        red on
*/

#include <Arduino.h>

const int GREEN_LED = 25;
const int YELLOW_LED = 26;
const int RED_LED = 27;
const int BUZZER = 14;

void handleCommand(String command);
void setOutputs(bool green, bool yellow, bool red, bool buzzer);

void setup() {
  Serial.begin(115200);

  pinMode(GREEN_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);

  handleCommand("NORMAL");
  Serial.println("ESP32 traffic light ready");
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();
  handleCommand(command);
}

void handleCommand(String command) {
  if (command == "NORMAL") {
    setOutputs(true, false, false, false);
  } else if (command == "SLOW") {
    setOutputs(false, true, false, true);
  } else if (command == "PULL_RIGHT") {
    setOutputs(false, true, true, true);
  } else if (command == "STOP") {
    setOutputs(false, false, true, true);
  }
}

void setOutputs(bool green, bool yellow, bool red, bool buzzer) {
  digitalWrite(GREEN_LED, green ? HIGH : LOW);
  digitalWrite(YELLOW_LED, yellow ? HIGH : LOW);
  digitalWrite(RED_LED, red ? HIGH : LOW);
  digitalWrite(BUZZER, buzzer ? HIGH : LOW);
}
