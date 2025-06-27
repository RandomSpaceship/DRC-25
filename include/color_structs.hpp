#pragma once

#include <stdint.h>

#include "config.hpp"

struct rgb_t;
struct big_rgb_t;
struct hsv_t;

struct rgb_t
{
    union
    {
        uint8_t raw[3] = {0, 0, 0};
        struct
        {
            union
            {
                uint8_t r;
                uint8_t red;
            };
            union
            {
                uint8_t g;
                uint8_t green;
            };
            union
            {
                uint8_t b;
                uint8_t blue;
            };
        };
    };

    operator big_rgb_t() const;
};

struct big_rgb_t
{
    union
    {
        uint32_t raw[3] = {0, 0, 0};
        struct
        {
            union
            {
                uint32_t r;
                uint32_t red;
            };
            union
            {
                uint32_t g;
                uint32_t green;
            };
            union
            {
                uint32_t b;
                uint32_t blue;
            };
        };
    };

    explicit operator rgb_t() const;

    big_rgb_t operator<<(unsigned long bits) const;
    big_rgb_t operator>>(unsigned long bits) const;

    big_rgb_t operator+(const big_rgb_t data) const;
    big_rgb_t operator+=(big_rgb_t data);

    big_rgb_t operator-(const big_rgb_t data) const;
    big_rgb_t operator-=(big_rgb_t data);

    big_rgb_t operator*(const int32_t multiplier) const;
    big_rgb_t operator*=(const int32_t multiplier);

    big_rgb_t operator/(const int32_t divider) const;
    big_rgb_t operator/=(const int32_t divider);
};

struct hsv_t
{
    union
    {
        uint8_t raw[3] = {0, 0, 0};
        struct
        {
            union
            {
                uint8_t h;
                uint8_t hue;
            };
            union
            {
                uint8_t s;
                uint8_t sat;
                uint8_t saturation;
            };
            union
            {
                uint8_t v;
                uint8_t val;
                uint8_t value;
            };
        };
    };
};

rgb_t hsvToRgb(hsv_t hsv);
big_rgb_t rainbowHsvToRgb(hsv_t hsv);

void hsv2rgb_rainbow(const hsv_t& hsv, rgb_t& rgb);