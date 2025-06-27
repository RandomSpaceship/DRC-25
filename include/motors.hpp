#pragma once

#include <cstdint>

#define MOTOR_MAX_SPEED 10000

void motorsInit();

void motorsSetEnabled(bool enabled = true);

// takes -MOTOR_MAX_SPEED - +MOTOR_MAX_SPEED for each motor
void motorsSetSpeeds(int32_t left, int32_t right);