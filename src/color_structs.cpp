#include "color_structs.hpp"

#include "config.hpp"

#define MAX_VALUE ((2UL >> DITHERED_BIT_DEPTH) - 1)

rgb_t::operator big_rgb_t() const
{
    big_rgb_t rgb = {(uint8_t)(r << (DITHERED_BIT_DEPTH - 8)),
                     (uint8_t)(g << (DITHERED_BIT_DEPTH - 8)),
                     (uint8_t)(b << (DITHERED_BIT_DEPTH - 8))};
    return rgb;
}

big_rgb_t::operator rgb_t() const
{
    big_rgb_t tmp = (*this) >> (DITHERED_BIT_DEPTH - 8);
    return {(uint8_t)tmp.r, (uint8_t)tmp.g, (uint8_t)tmp.b};
}

big_rgb_t big_rgb_t::operator<<(unsigned long bits) const
{
    return {r << bits, g << bits, b << bits};
}
big_rgb_t big_rgb_t::operator>>(unsigned long bits) const
{
    return {r >> bits, g >> bits, b >> bits};
}

big_rgb_t big_rgb_t::operator+(const big_rgb_t data) const
{
    return {r + data.r, g + data.g, b + data.b};
}
big_rgb_t big_rgb_t::operator+=(big_rgb_t data)
{
    r += data.r;
    g += data.g;
    b += data.b;

    return *this;
}

big_rgb_t big_rgb_t::operator-(const big_rgb_t data) const
{
    uint32_t tmpr, tmpg, tmpb;
    if (r < data.r)
        tmpr = 0;
    else
        tmpr = r - data.r;

    if (g < data.g)
        tmpg = 0;
    else
        tmpg = g - data.g;

    if (b < data.b)
        tmpb = 0;
    else
        tmpb = b - data.b;

    return {tmpr, tmpg, tmpb};
}
big_rgb_t big_rgb_t::operator-=(big_rgb_t data)
{
    if (r < data.r)
        r = 0;
    else
        r -= data.r;
    if (g < data.g)
        g = 0;
    else
        g -= data.g;
    if (b < data.b)
        b = 0;
    else
        b -= data.b;

    return *this;
}

big_rgb_t big_rgb_t::operator*(const int32_t multiplier) const
{
    return {r * multiplier, g * multiplier, b * multiplier};
}
big_rgb_t big_rgb_t::operator*=(const int32_t multiplier)
{
    r *= multiplier;
    g *= multiplier;
    b *= multiplier;
    return *this;
}

big_rgb_t big_rgb_t::operator/(const int32_t divider) const
{
    return {r / divider, g / divider, b / divider};
}
big_rgb_t big_rgb_t::operator/=(const int32_t divider)
{
    r /= divider;
    g /= divider;
    b /= divider;
    return *this;
}

rgb_t hsvToRgb(hsv_t hsv)
{
    rgb_t rgb;
    uint8_t region, p, q, t;
    uint16_t h, s, v, remainder;

    if (hsv.s == 0)
    {
        rgb.r = hsv.v;
        rgb.g = hsv.v;
        rgb.b = hsv.v;
        return rgb;
    }

    // converting to 16 bit to prevent overflow
    h = hsv.h;
    s = hsv.s;
    v = hsv.v;

    region = h / 43;
    remainder = (h - (region * 43)) * 6;

    p = (v * (255 - s)) >> 8;
    q = (v * (255 - ((s * remainder) >> 8))) >> 8;
    t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

    switch (region)
    {
    case 0:
        rgb.r = v;
        rgb.g = t;
        rgb.b = p;
        break;
    case 1:
        rgb.r = q;
        rgb.g = v;
        rgb.b = p;
        break;
    case 2:
        rgb.r = p;
        rgb.g = v;
        rgb.b = t;
        break;
    case 3:
        rgb.r = p;
        rgb.g = q;
        rgb.b = v;
        break;
    case 4:
        rgb.r = t;
        rgb.g = p;
        rgb.b = v;
        break;
    default:
        rgb.r = v;
        rgb.g = p;
        rgb.b = q;
        break;
    }

    return rgb;
}

// big_rgb_t lookup_table[] = {{MAX_VALUE, 0, 0}, {MAX_VALUE / 2, MAX_VALUE / 2, 0}, {0, MAX_VALUE, 0}, {0, MAX_VALUE / 2, MAX_VALUE / 2}, {0, 0, MAX_VALUE}, {MAX_VALUE / 2, 0, MAX_VALUE / 2}};  // 0-255 evenly spaced
int hueTable[] = {0, 21, 43, 85, 128, 171, 192, 213};
#include <stdio.h>
big_rgb_t rainbowHsvToRgb(hsv_t hsv)
{
    uint8_t hue = hsv.h;
    uint8_t seg = hue >> 5;
    uint8_t lerp = hue & (0b00011111);

    uint32_t tableHue = hueTable[seg];
    uint32_t nextHue = seg == 7 ? 256 : hueTable[seg + 1];

    // printf("%d segment %d\n", lerp, seg);
    hue = ((nextHue * lerp) + (tableHue * (31 - lerp))) / 31;

    big_rgb_t color = hsvToRgb({hue, hsv.s, hsv.v});

    // color *= hsv.v;
    // color /= 255;

    return color;
}