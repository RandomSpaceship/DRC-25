#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>

#include "WS2812.pio.h"
#include "color_structs.hpp"
#include "config.hpp"
#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/i2c.h"
#include "hardware/pio.h"
#include "hardware/pwm.h"
#include "hardware/structs/ioqspi.h"
#include "hardware/structs/sio.h"
#include "hardware/sync.h"
#include "hardware/uart.h"
#include "pico/bootrom.h"
#include "pico/multicore.h"
#include "pico/mutex.h"
#include "pico/stdlib.h"
#include "servo.pio.h"

#define MOTOR_A1B_PIN 18
#define MOTOR_A1A_PIN 19
#define MOTOR_B1B_PIN 20
#define MOTOR_B1A_PIN 21

#define SLOTS_PER_REV 20

// Item Collection
#define LIFT_MOTOR_AIN1_PIN 14
#define LIFT_MOTOR_AIN2_PIN 15
#define SERVO_MOTOR_PIN     3
#define LIMIT_SWITCH_PIN    2

mutex ledDataMutex;
std::atomic_bool ledDataAvailable;
big_rgb_t ledDataBuffer[LED_COUNT];
const uint8_t LED_TX_PIN = 17;

float leftAdcReading = 0;
float rightAdcReading = 0;

using namespace std::chrono_literals;
using namespace std::chrono;

// int leftEncoderCount = 0;
// int rightEncoderCount = 0;

steady_clock::duration leftAvgDuration = 0ms;
steady_clock::duration rightAvgDuration = 0ms;

#define PWM_COUNT           2000
#define ENC_AVG_COUNT       2
#define ENC_SPEED_AVG_COUNT 4
#define ENC_HIGH            1000
#define ENC_LOW             500
static steady_clock::time_point lastLeftUpdate = steady_clock::now();
static steady_clock::time_point lastRightUpdate = steady_clock::now();

float leftClicksPerSec = 0;
float rightClicksPerSec = 0;

bool leftSpinning = false;
bool rightSpinning = false;

#define MOTOR_STOP_MULTIPLIER 2.5

int servo_pio_sm = 1;
PIO servo_pio = pio0;
// Write `period` to the input shift register
void pio_servo_pwm_set_period(PIO pio, uint sm, uint32_t period)
{
    pio_sm_set_enabled(pio, sm, false);
    pio_sm_put_blocking(pio, sm, period);
    pio_sm_exec(pio, sm, pio_encode_pull(false, false));
    pio_sm_exec(pio, sm, pio_encode_out(pio_isr, 32));
    pio_sm_set_enabled(pio, sm, true);
}

// Write `level` to TX FIFO. State machine will copy this into X.
void pio_servo_pwm_set_level(PIO pio, uint sm, uint32_t level)
{
    pio_sm_put_blocking(pio, sm, level);
}

template <typename T, int N>
class MovingAvg
{
public:
    T add(T value)
    {
        _accumulator -= _values[_index];
        _accumulator += value;

        _values[_index++] = value;
        _index %= N;

        _avg = _accumulator / N;
        return get();
    }
    T get()
    {
        return _avg;
    }

    void reset(T value)
    {
        _accumulator = value * N;
        for (int i = 0; i < N; i++)
        {
            _values[i] = value;
        }
        _avg = value;
    }

private:
    T _values[N] = {};
    int _index = 0;
    T _accumulator = {};
    T _avg = {};
};

class MotorController
{
public:                    // 0.2/0.03 working
    const float p = 2.00;  // 0.35
    const float i = 0.13;
    const float d = -0.25;  // 0.03

    void reset()
    {
        _prevInput = 0;
        _targetClicksPerSec = 0;
        _derivAvg.reset(0);
        _acc = 0;
        _accCounter = 0;
        _integralAvg.reset(0);
    }

    void setSetpoint(float targetRevPerMin)
    {
        _targetClicksPerSec = targetRevPerMin;  // TODO: TESTING ONLY
        return;
        float slotsPerMin = targetRevPerMin * SLOTS_PER_REV;
        float changesPerMin = slotsPerMin * 2;  // rising AND falling edge - TODO: should this be changed?
        float changesPerSec = changesPerMin / 60;
        _targetClicksPerSec = changesPerSec;
    }
    float getSetpoint()
    {
        return _targetClicksPerSec;
    }

    // this run at 1s/20ms = 50Hz
    float update(const float currentClicksPerSec)
    {
        float error = _targetClicksPerSec - currentClicksPerSec;

        float derivative = _derivAvg.add(currentClicksPerSec - _prevInput);
        _acc += error;
        _accCounter++;

        float integral = _acc / _accCounter;

        _prevInput = currentClicksPerSec;

        if (_targetClicksPerSec == 0)
        {
            return 0;
        }

        float sum = (error * p) + (integral * i) + (derivative * d);
        // printf("%06.2f %06.2f %06.2f   %06.2f  ", error, integral, derivative, sum);
        return sum;
    }

private:
    float _acc = 0;
    int _accCounter = 0;
    MovingAvg<float, 3> _derivAvg;
    MovingAvg<float, 400> _integralAvg;
    float _prevInput = 0;
    float _targetClicksPerSec = 0;
};

void updateAdcs()
{
    static bool firstRun = true;
    static bool prevLeftEncoderState = false;
    static bool prevRightEncoderState = false;

    static bool leftEncoderState = false;
    static bool rightEncoderState = false;

    static MovingAvg<float, ENC_AVG_COUNT> leftAdcAvg;
    static MovingAvg<float, ENC_AVG_COUNT> rightAdcAvg;

    static MovingAvg<steady_clock::duration, ENC_SPEED_AVG_COUNT> leftDurationMovingAvg;
    static MovingAvg<steady_clock::duration, ENC_SPEED_AVG_COUNT> rightDurationMovingAvg;

    static steady_clock::duration currentLeftDuration = 0ms;
    static steady_clock::duration currentRightDuration = 0ms;

    const auto now = steady_clock::now();

    if (firstRun)
    {
        firstRun = false;
        leftAdcAvg.reset(0);
        rightAdcAvg.reset(0);
        leftDurationMovingAvg.reset(10s);
        rightDurationMovingAvg.reset(10s);
    }

    currentLeftDuration = now - lastLeftUpdate;
    currentRightDuration = now - lastRightUpdate;

    adc_select_input(1);
    leftAdcReading = leftAdcAvg.add(adc_read());

    if (leftAdcReading > ENC_HIGH)
    {
        leftEncoderState = true;
    }
    if (leftAdcReading < ENC_LOW)
    {
        leftEncoderState = false;
    }

    if (prevLeftEncoderState != leftEncoderState)
    {
        prevLeftEncoderState = leftEncoderState;
        lastLeftUpdate = now;
        leftAvgDuration = leftDurationMovingAvg.add(currentLeftDuration);
        currentLeftDuration = 0ms;

        // printf("L\r\n");
    }

    adc_select_input(0);
    rightAdcReading = rightAdcAvg.add(adc_read());

    if (rightAdcReading > ENC_HIGH)
    {
        rightEncoderState = true;
    }
    if (rightAdcReading < ENC_LOW)
    {
        rightEncoderState = false;
    }

    if (prevRightEncoderState != rightEncoderState)
    {
        prevRightEncoderState = rightEncoderState;
        lastRightUpdate = now;
        rightAvgDuration = rightDurationMovingAvg.add(currentRightDuration);
        currentRightDuration = 0ms;

        // printf("R\r\n");
    }

    leftSpinning = (currentLeftDuration < leftAvgDuration * MOTOR_STOP_MULTIPLIER) || (currentLeftDuration < 50ms);
    rightSpinning = (currentRightDuration < rightAvgDuration * MOTOR_STOP_MULTIPLIER) || (currentRightDuration < 50ms);

    long rightDuration = duration_cast<microseconds>(rightAvgDuration).count();
    long leftDuration = duration_cast<microseconds>(leftAvgDuration).count();

    // printf("%08d %08d\r\n", leftDuration, rightDuration);
    if (leftSpinning && leftDuration)
    {
        leftClicksPerSec = 1000000.0 / leftDuration;
    }
    else if (leftClicksPerSec)
    {
        leftDurationMovingAvg.reset(0ms);
        if (currentLeftDuration < 500ms)
        {
            leftClicksPerSec = 1000000.0 / duration_cast<microseconds>(currentLeftDuration).count();
        }
        else
        {
            // printf("LL\r\n");
            leftClicksPerSec = 0;
        }
    }
    if (rightSpinning && rightDuration)
    {
        rightClicksPerSec = 1000000.0 / rightDuration;
    }
    else if (rightClicksPerSec)
    {
        rightDurationMovingAvg.reset(0ms);
        if (currentRightDuration < 500ms)
        {
            rightClicksPerSec = 1000000.0 / duration_cast<microseconds>(currentRightDuration).count();
        }
        else
        {
            // printf("RR\r\n");
            rightClicksPerSec = 0;
        }
    }
}

void core1_thread()
{
    {
        uint offset = pio_add_program(pio0, &ws2812_program);

        printf("Init PIO\r\n");
        ws2812_program_init(pio0, 0, offset, LED_TX_PIN, LED_CLOCK_SPEED, false);
    }

    printf("Begin LED loop\r\n");
    while (1)
    {
        static big_rgb_t savedLedDataBuffer[LED_COUNT]{};
        static big_rgb_t ledErrorBuffer[LED_COUNT]{};

        if (ledDataAvailable.load())
        {
            mutex_enter_blocking(&ledDataMutex);
            ledDataAvailable.store(false);
            memcpy(savedLedDataBuffer, ledDataBuffer, LED_COUNT * sizeof(big_rgb_t));
            mutex_exit(&ledDataMutex);
        }

        for (int i = 0; i < LED_COUNT; i++)
        {
            big_rgb_t total = (savedLedDataBuffer[i] + ledErrorBuffer[i]);
            rgb_t output = (rgb_t)total;
            ledErrorBuffer[i] = (total - (big_rgb_t)output);

            const uint32_t raw_data = output.r << 8u | output.g << 16u | output.b << 0u;
            // printf("Put %d %d\r\n", 0, i);
            pio_sm_put_blocking(pio0, 0, raw_data << 8);
            // printf("Finish put %d %d\r\n", 0, i);
        }
        sleep_us(1000);
    }
}

void setLiftMotorSpeed(int speed)
{
    if (speed >= PWM_COUNT)
        speed = PWM_COUNT;
    if (speed <= -PWM_COUNT)
        speed = -(PWM_COUNT);

    if (speed > 0)
    {
        pwm_set_gpio_level(LIFT_MOTOR_AIN1_PIN, speed);
        pwm_set_gpio_level(LIFT_MOTOR_AIN2_PIN, 0);
    }
    else
    {
        pwm_set_gpio_level(LIFT_MOTOR_AIN1_PIN, 0);
        pwm_set_gpio_level(LIFT_MOTOR_AIN2_PIN, -speed);
    }
}

void stopLift()
{
    pwm_set_gpio_level(LIFT_MOTOR_AIN1_PIN, 0);
    pwm_set_gpio_level(LIFT_MOTOR_AIN2_PIN, 0);
}

// +/-PWM_COUNT for both
void setMotorSpeeds(int left, int right)
{
    if (left >= PWM_COUNT)
        left = PWM_COUNT;
    if (left <= -PWM_COUNT)
        left = -(PWM_COUNT);

    if (right >= PWM_COUNT)
        right = PWM_COUNT;
    if (right <= -PWM_COUNT)
        right = -(PWM_COUNT);

    if (right > 0)
    {
        pwm_set_gpio_level(MOTOR_A1B_PIN, right);
        pwm_set_gpio_level(MOTOR_A1A_PIN, 0);
    }
    else
    {
        pwm_set_gpio_level(MOTOR_A1B_PIN, 0);
        pwm_set_gpio_level(MOTOR_A1A_PIN, -right);
    }

    if (left > 0)
    {
        pwm_set_gpio_level(MOTOR_B1B_PIN, left);
        pwm_set_gpio_level(MOTOR_B1A_PIN, 0);
    }
    else
    {
        pwm_set_gpio_level(MOTOR_B1B_PIN, 0);
        pwm_set_gpio_level(MOTOR_B1A_PIN, -left);
    }
}

void setGripperPos(int degrees)
{
    static uint servo_offset = pio_add_program(servo_pio, &servo_program);
    if (degrees == 255)
    {
        printf("Disabling servo...\r\n");
        pio_servo_pwm_set_level(servo_pio, servo_pio_sm, 4000);
        // gpio_init(SERVO_MOTOR_PIN);
        // gpio_put(SERVO_MOTOR_PIN, 0);
        return;
    }
    else
    {
        printf("Writing servo pos %d\r\n", degrees);
        servo_program_init(servo_pio, servo_pio_sm, servo_offset, SERVO_MOTOR_PIN);
        pio_servo_pwm_set_period(servo_pio, servo_pio_sm, 4000);
    }
    if (degrees > 150)
        degrees = 150;
    else if (degrees < 10)
        degrees = 10;
    int pwmOffset = (degrees * 2000) / 180;
    if (pwmOffset < 0)
        pwmOffset = 0;
    if (pwmOffset > 2000)
        pwmOffset = 2000;
    int finalPwm = pwmOffset + 500;

    pio_servo_pwm_set_level(servo_pio, servo_pio_sm, finalPwm);
    // pwm_set_gpio_level(SERVO_MOTOR_PIN, finalPwm);
}

enum led_state
{
    LED_RED = 0,
    LED_YELLOW,
    LED_GREEN,
    LED_RAINBOW
};

#pragma pack(push, 1)
struct serial_data_t
{
    uint8_t startByte;
    uint8_t ledState;
    int16_t leftClicksPerSec;
    int16_t rightClicksPerSec;
    uint16_t gripperLiftTime;
    uint8_t gripperClosePosition;
};
#pragma pack(pop)

int maxLiftTime = 2000;

char serialBuff[100];
int serialBuffIndex = 0;

MotorController leftController;
MotorController rightController;

const int homingPwm = -950;
const int liftingPwm = 1850;

int main()
{
    static float prevLeftSpeed = 0;
    static float prevRightSpeed = 0;
    led_state ledState = LED_RAINBOW;
    auto startup = steady_clock::now();
    static auto lastSerial = steady_clock::now();
    static serial_data_t data{};
    // clock init
    // set_sys_clock_48mhz();
    // set_sys_clock_khz(270 * 1000, false);

    gpio_set_dir(PICO_DEFAULT_LED_PIN, true);
    gpio_put(PICO_DEFAULT_LED_PIN, true);

    // Analog init
    adc_init();
    adc_gpio_init(26);
    adc_gpio_init(27);

    // IO init
    stdio_init_all();
    static bool rightEnabled = true;
    static bool leftEnabled = true;
    static bool prevLeftStopped = false;
    static bool prevRightStopped = false;

    static int prevLiftTime = 0;
    static auto motorMoveStart = steady_clock::now();
    static bool shouldHome = true;

    printf("\nBoot\r\n");
    sleep_ms(100);  // prevents weird div by zero errors with the timings
    // sleep_ms(8000);  // wait for terminal to connect
    // getchar();

    // Threading init
    mutex_init(&ledDataMutex);

    multicore_launch_core1(core1_thread);

    leftController.setSetpoint(0);
    rightController.setSetpoint(0);

    const uint8_t LED_BRIGHTNESS = 80;
    static hsv_t hsv{0, 255, LED_BRIGHTNESS};
    static uint8_t hue = 0;
    gpio_init(MOTOR_A1A_PIN);
    gpio_init(MOTOR_A1B_PIN);
    gpio_init(MOTOR_B1A_PIN);
    gpio_init(MOTOR_B1B_PIN);
    // Item Collection
    gpio_init(LIFT_MOTOR_AIN1_PIN);
    gpio_init(LIFT_MOTOR_AIN2_PIN);

    gpio_set_dir(MOTOR_A1A_PIN, GPIO_OUT);
    gpio_set_dir(MOTOR_A1B_PIN, GPIO_OUT);
    gpio_set_dir(MOTOR_B1A_PIN, GPIO_OUT);
    gpio_set_dir(MOTOR_B1B_PIN, GPIO_OUT);
    // Item Collection
    gpio_set_dir(LIFT_MOTOR_AIN1_PIN, GPIO_OUT);
    gpio_set_dir(LIFT_MOTOR_AIN2_PIN, GPIO_OUT);

    // set up pwm
    gpio_set_function(MOTOR_A1A_PIN, GPIO_FUNC_PWM);
    gpio_set_function(MOTOR_A1B_PIN, GPIO_FUNC_PWM);
    gpio_set_function(MOTOR_B1A_PIN, GPIO_FUNC_PWM);
    gpio_set_function(MOTOR_B1B_PIN, GPIO_FUNC_PWM);
    // Item Collection
    gpio_set_function(LIFT_MOTOR_AIN1_PIN, GPIO_FUNC_PWM);
    gpio_set_function(LIFT_MOTOR_AIN2_PIN, GPIO_FUNC_PWM);

    uint16_t SLICE_COUNT = PWM_COUNT - 1;
    printf("pwm slice: %d\r\n", SLICE_COUNT);
    pwm_set_wrap(pwm_gpio_to_slice_num(MOTOR_A1A_PIN), SLICE_COUNT);
    pwm_set_wrap(pwm_gpio_to_slice_num(MOTOR_B1A_PIN), SLICE_COUNT);
    // Item Collection
    pwm_set_wrap(pwm_gpio_to_slice_num(LIFT_MOTOR_AIN1_PIN), SLICE_COUNT);

    // Set the PWM running
    pwm_set_clkdiv(pwm_gpio_to_slice_num(MOTOR_A1A_PIN), 40.0f);
    pwm_set_clkdiv(pwm_gpio_to_slice_num(MOTOR_B1A_PIN), 40.0f);
    pwm_set_enabled(pwm_gpio_to_slice_num(MOTOR_A1A_PIN), true);
    pwm_set_enabled(pwm_gpio_to_slice_num(MOTOR_B1A_PIN), true);
    // Item Collection Motor
    pwm_set_clkdiv(pwm_gpio_to_slice_num(LIFT_MOTOR_AIN1_PIN), 40.0f);
    pwm_set_enabled(pwm_gpio_to_slice_num(LIFT_MOTOR_AIN1_PIN), true);
    gpio_set_dir(LIMIT_SWITCH_PIN, GPIO_IN);
    gpio_set_pulls(LIMIT_SWITCH_PIN, true, false);

    setMotorSpeeds(0, 0);

    setGripperPos(10);

    stopLift();
    sleep_ms(500);
    setGripperPos(255);
    // setGripperPos(120);
    // setLiftMotorSpeed(-600);
    // sleep_ms(400);
    // stopLift();
    // reset_usb_boot(0, 0);

    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    steady_clock::time_point lastUpdate = steady_clock::now();

    steady_clock::time_point lastMotorUpdate = steady_clock::now();

    float targetLeftSpeed = 0;
    float targetRightSpeed = 0;

    float currentLeftPwm = 0;
    float currentRightPwm = 0;

    // while ((time_us_64() / 1000) < 1200000)
    while (true)
    {
        if (steady_clock::now() - lastUpdate > 50ms)
        {
            lastUpdate = steady_clock::now();
            mutex_enter_blocking(&ledDataMutex);
            for (int i = 0; i < LED_COUNT; i++)
            {
                switch (ledState)
                {
                default:
                case LED_RED:
                    ledDataBuffer[i] = rgb_t{LED_BRIGHTNESS, 0, 0};
                    break;
                case LED_YELLOW:
                    ledDataBuffer[i] = rgb_t{LED_BRIGHTNESS, LED_BRIGHTNESS, 0};
                    break;
                case LED_GREEN:
                    ledDataBuffer[i] = rgb_t{0, LED_BRIGHTNESS, 0};
                    break;
                case LED_RAINBOW:
                    hsv.h = hue + i * 15;
                    ledDataBuffer[i] = rainbowHsvToRgb(hsv);
                    break;
                }
            }
            ledDataAvailable.store(true);
            mutex_exit(&ledDataMutex);
            hue += 10;

            // printf("%07.2f %07.2f\r\n", leftClicksPerSec, rightClicksPerSec);
        }
        if (steady_clock::now() - lastMotorUpdate > 20ms)
        {
            lastMotorUpdate = steady_clock::now();

            leftController.setSetpoint(abs(targetLeftSpeed));
            rightController.setSetpoint(abs(targetRightSpeed));

            float leftOut = (leftEnabled) ? leftController.update(leftClicksPerSec) : 0;
            float rightOut = (rightEnabled) ? rightController.update(rightClicksPerSec) : 0;
            // printf("%05d %05d    %05d %05d     %05d\r\n", leftMs, leftCurMs, rightMs, rightCurMs, time);

            if (targetLeftSpeed < 0 && prevLeftSpeed > 0)
            {
                currentLeftPwm = 0;
            }
            else if (targetLeftSpeed > 0 && prevLeftSpeed < 0)
            {
                currentLeftPwm = 0;
            }
            else if (targetLeftSpeed == 0)
            {
                // leftController.reset();
                leftOut = 0;
                currentLeftPwm = 0;
            }
            prevLeftSpeed = targetLeftSpeed;

            if (targetRightSpeed < 0 && prevRightSpeed > 0)
            {
                currentRightPwm = 0;
            }
            else if (targetRightSpeed > 0 && prevRightSpeed < 0)
            {
                currentRightPwm = 0;
            }
            else if (targetRightSpeed == 0)
            {
                // rightController.reset();
                rightOut = 0;
                currentRightPwm = 0;
            }
            prevRightSpeed = targetRightSpeed;

            currentRightPwm += rightOut;
            if (!rightEnabled)
                currentRightPwm = 0;
            else if (currentRightPwm < 0)
                currentRightPwm = 0;
            else if (currentRightPwm > PWM_COUNT)
                currentRightPwm = PWM_COUNT;

            currentLeftPwm += leftOut;
            if (!leftEnabled)
                currentLeftPwm = 0;
            else if (currentLeftPwm < 0)
                currentLeftPwm = 0;
            else if (currentLeftPwm > PWM_COUNT)
                currentLeftPwm = PWM_COUNT;

            if (!prevLeftStopped && !leftSpinning)
            {
                currentLeftPwm += 100;
            }
            if (!prevRightStopped && !rightSpinning)
            {
                currentRightPwm += 100;
            }
            prevLeftStopped = !leftSpinning;
            prevRightStopped = !rightSpinning;

            int l = (int)round(currentLeftPwm);
            int r = (int)round(currentRightPwm);

            if (!leftSpinning && leftEnabled && leftOut)
                l += 100;
            if (!rightSpinning && rightEnabled && rightOut)
                r += 100;

            if (leftEnabled && targetLeftSpeed)
                l += 540;
            if (rightEnabled && targetRightSpeed)
                r += 540;

            if (targetLeftSpeed < 0)
                l = -l;
            if (targetRightSpeed < 0)
                r = -r;
            setMotorSpeeds(l, r);

            // printf("%07.1f    %06.2f   %04d   %06.0f\r\n", leftClicksPerSec, leftOut, l, leftAdcReading);
            printf("%07.1f %07.1f   %06.2f %06.2f   %04d %04d   %06.0f %06.0f\r\n", leftClicksPerSec, rightClicksPerSec, leftOut, rightOut, l, r, leftAdcReading, rightAdcReading);
        }
        updateAdcs();

        // sleep_ms(10);
        // if (get_bootsel_button())
        // {
        //     break;
        // }
        if (char c = getchar_timeout_us(0); c != PICO_ERROR_TIMEOUT)
        {
            serialBuff[serialBuffIndex] = c;
            if (serialBuffIndex == 0)
            {
                // if (c == 'q')
                // {
                //     break;
                // }
                if (c == 0xAA)
                {
                    serialBuffIndex++;
                }
            }
            else
            {
                if (serialBuffIndex == sizeof(serial_data_t))
                {
                    printf("SER REC\r\n");
                    serialBuffIndex = 0;
                    serial_data_t *pData = (serial_data_t *)serialBuff;
                    if (pData->gripperLiftTime > maxLiftTime)
                        pData->gripperLiftTime = maxLiftTime;
                    data = *pData;
                    memcpy(&data, pData, sizeof(serial_data_t));

                    ledState = (led_state)pData->ledState;
                    if (!targetLeftSpeed && pData->leftClicksPerSec)
                    {
                        currentLeftPwm += 200;
                    }
                    if (!targetRightSpeed && pData->rightClicksPerSec)
                    {
                        currentRightPwm += 200;
                    }
                    targetLeftSpeed = pData->leftClicksPerSec / 10.0f;
                    targetRightSpeed = pData->rightClicksPerSec / 10.0f;
                    lastSerial = steady_clock::now();
                    if ((data.gripperLiftTime != prevLiftTime) || (data.gripperLiftTime == 0))
                        shouldHome = true;
                    setGripperPos(data.gripperClosePosition);
                }
                else
                {
                    serialBuff[serialBuffIndex++] = c;
                }
            }
        }

        if ((steady_clock::now() - lastSerial > 500ms) && (targetLeftSpeed || targetRightSpeed || (ledState != LED_RAINBOW)))
        {
            ledState = LED_RAINBOW;
            serialBuffIndex = 0;
            targetLeftSpeed = 0;
            targetRightSpeed = 0;
            data.gripperClosePosition = 255;
            setGripperPos(255);
        }

        // Item Collection Code:

        const auto now = steady_clock::now();
        if (shouldHome)
        {
            // limit switch pressed (low)
            if (!gpio_get(LIMIT_SWITCH_PIN))
            {
                // stop motor
                stopLift();
                shouldHome = false;
                prevLiftTime = 0;
            }
            else
            {
                setLiftMotorSpeed(homingPwm);
            }
        }
        else
        {
            if (prevLiftTime != data.gripperLiftTime)
            {
                prevLiftTime = data.gripperLiftTime;
                motorMoveStart = now;
            }

            if (now - motorMoveStart < milliseconds(data.gripperLiftTime))
            {
                setLiftMotorSpeed(liftingPwm);
            }
            else
            {
                stopLift();
            }
        }
    }

    gpio_put(PICO_DEFAULT_LED_PIN, false);

    sleep_ms(10);
    printf("Finished\n");
    fflush(stdout);
    reset_usb_boot(0, 0);
}  //*/