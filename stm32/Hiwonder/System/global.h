#ifndef __GLOBAL_H_
#define __GLOBAL_H_

#include "global_conf.h"

#include "SEGGER_RTT.h"
#include "main.h"

#include "stm32f4xx_hal.h"

#include "i2c.h"
#include "spi.h"
#include "usart.h"
#include "dma.h"
#include "tim.h"
#include "lwrb.h"
#include "lwmem.h"

#include "led.h"
#include "encoder_motor.h"
#include "pid.h"
#include "chassis.h"
#include "rgb_spi.h"

// 全系统全局变量
extern LEDObjectTypeDef *leds[LED_NUM];
extern EncoderMotorObjectTypeDef *motors[4];
extern ChassisTypeDef *chassis;

void set_chassis_type(uint8_t chassis_type);
void change_battery_limit(uint16_t limit);

#endif

