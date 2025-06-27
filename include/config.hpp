#pragma once

#define MAX_BRIGHTNESS 255

#define LED_COUNT 8

// *technically* the LEDs clock in data at 800kHz but 1.3MHz works fine somehow, so clocked at 1.2MHz for "safety margin"
#define LED_CLOCK_SPEED 800000UL
// led_clock / (bits_per_led * led_count * (2 ^ (DITHERED_BIT_DEPTH - 8))) = 52Hz update rate for all panels @ 800kHz
// however, "overclocked" to 1.2MHz, the refresh rate becomes 78Hz
#define DITHERED_BIT_DEPTH 8UL