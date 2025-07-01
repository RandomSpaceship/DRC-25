/**
 * @file app.c
 * @author Lu Yongping (Lucas@hiwonder.com)
 * @brief 主应用逻辑(main appilication logic)
 * @version 0.1
 * @date 2023-05-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "cmsis_os2.h"
#include "led.h"
#include "lwmem_porting.h"
#include "adc.h"
#include "rgb_spi.h"
#include "motors_param.h"
#include "FreeRTOS.h"
#include "task.h"
#include "main.h"
#include "cmsis_os.h"
#include "queue.h"

#include "encoder_motor.h"
#include "mecanum_chassis.h"
#include "usart.h"

void leds_init(void);
void motors_init(void);
void chassis_init(void);

#pragma pack(push, 1)
typedef struct {
	int16_t fwd;
	int16_t strafe;
	int16_t turn;
} motor_data_t;
#pragma pack(pop)

extern EncoderMotorObjectTypeDef *motors[4];
static int16_t conv_pulse(int32_t in) {
//	in *= 1000;
	if (in > 1000)
		in = 1000;
	else if (in < -1000)
		in = -1000;
	return in;
}
static void set_motors(int32_t rps_lh, int32_t rps_lt, int32_t rps_rh,
		int32_t rps_rt) {
//	encoder_motor_set_speed(motors[2], rps_lh);
//	encoder_motor_set_speed(motors[0], -rps_lt);
//	encoder_motor_set_speed(motors[3], rps_rh);
//	encoder_motor_set_speed(motors[1], rps_rt);
	motors[2]->set_pulse(NULL, conv_pulse(-rps_lh));
	motors[0]->set_pulse(NULL, conv_pulse(-rps_lt));
	motors[3]->set_pulse(NULL, conv_pulse(rps_rh));
	motors[1]->set_pulse(NULL, conv_pulse(rps_rt));
}

typedef StaticQueue_t osStaticMessageQDef_t;
typedef StaticTimer_t osStaticTimerDef_t;
osMessageQueueId_t motorDataQueueHandle;
uint8_t motorDataQueueBuffer[8 * sizeof(motor_data_t)];
osStaticMessageQDef_t motorDataQueueControlBlock;
const osMessageQueueAttr_t motorDataQueueAttributes = { .name = "motor_data",
		.cb_mem = &motorDataQueueControlBlock, .cb_size =
				sizeof(motorDataQueueControlBlock), .mq_mem =
				&motorDataQueueBuffer, .mq_size = sizeof(motorDataQueueBuffer) };

static void set_mecanum_speed(int32_t fwd, int32_t strafe, int32_t turn) {
	int32_t frontRight = fwd + strafe - turn;
	int32_t frontLeft = fwd - strafe + turn;
	int32_t rearRight = fwd - strafe - turn;
	int32_t rearLeft = fwd + strafe + turn;
	set_motors(frontLeft, rearLeft, frontRight, rearRight);
}

uint8_t connected = 0;
void app_task_entry(void *argument) {
	osDelay(500);
//	extern osTimerId_t led_timerHandle;
//	extern osTimerId_t buzzer_timerHandle;
//	extern osTimerId_t button_timerHandle;
//	extern osTimerId_t battery_check_timerHandle;
//	extern osMessageQueueId_t moving_ctrl_queueHandle;

	leds_init();
	motors_init();
	WS2812b_Configuration();

//	set_motor_param(motor, tpc, rps_limit, kp, ki, kd)
//	set_chassis_type(CHASSIS_TYPE_JETAUTO);

//	osTimerStart(led_timerHandle, LED_TASK_PERIOD);
//	osTimerStart(buzzer_timerHandle, BUZZER_TASK_PERIOD);
//	osTimerStart(battery_check_timerHandle, BATTERY_TASK_PERIOD);
//	packet_handle_init();

//    osDelay(50);

//    char msg = '\0';
//    uint8_t msg_prio;
//    osMessageQueueReset(moving_ctrl_queueHandle);

//    led_on(leds[0]);
//        led_off(leds[1]);
//        led_flash(leds[2] , 100 , 1000 , 0);
//    uint8_t rgb[6] = {0 , 0 , 0,0,0,0};
//        rgb[0] = 0;
//        rgb[1] = 0;
//        rgb[2] = 0xFF;
//        set_id_rgb_color(1,rgb);
//	osDelay(100);
//    set_rgb_color(rgb);
	uint8_t rgb[3] = { 0, 0, 250 };
//    set_id_rgb_color(0,rgb);
//    osDelay(500);

//	MecanumChassisTypeDef chassis = { .wheelbase = 1, .shaft_length = 0,
//			.wheel_diameter = 1, .correction_factor = 1000, .set_motors =
//					set_motors };
//	mecanum_chassis_object_init(&chassis);

	TickType_t lastBlink = xTaskGetTickCount();
	uint8_t blink = 0;

	while (1) {
		TickType_t now = xTaskGetTickCount();
		if (now - lastBlink >= 300) {
			lastBlink = now;
			blink = !blink;
			if (!connected) {
				rgb[0] = 0;
				rgb[1] = blink ? 255 : 0;
				rgb[2] = 0;
			} else {
				rgb[0] = blink ? 255 : 0;
				rgb[1] = 0;
				rgb[2] = 0;
			}
			set_id_rgb_color(0, rgb);
		}

		motor_data_t data;
		if (xQueueReceive(motorDataQueueHandle, &data, 10) == pdPASS) {
			set_mecanum_speed(data.fwd, data.strafe, data.turn);
//			printf("fwd: %05d strafe: %05d turn: %05d\r\n", (int) data.fwd,
//					(int) data.strafe, (int) data.turn);
		}
	}
}

#define RBUF_LEN 512
static char rxBuf[RBUF_LEN];
size_t rxHead = 0;
size_t rxTail = 0;

static uint8_t hasChars(void) {
	return rxHead != rxTail;
}
static char getChar(void) {
	if (!hasChars())
		return 0;
	char c = rxBuf[rxTail];
	rxTail++;
	if (rxTail >= RBUF_LEN)
		rxTail = 0;
	return c;
}
static void startRec() {
	HAL_UART_Receive_IT(&huart1, (uint8_t*) (rxBuf + rxHead), 1);
}
static void rxcbComplete(UART_HandleTypeDef *huart) {
	rxHead++;
	if (rxHead >= RBUF_LEN)
		rxHead = 0;
	startRec();
}

static char txBuf[RBUF_LEN];
size_t txHead = 0;
size_t txTail = 0;

static void startTransmit(void) {
	HAL_UART_Transmit_IT(&huart1, (uint8_t*) &txBuf[txTail], 1);
}
void transmitPrintf(char c) {
	txBuf[txHead++] = c;
	if (txHead >= RBUF_LEN)
		txHead = 0;
	startTransmit();
}
static void txcbComplete(UART_HandleTypeDef *huart) {
	txTail++;
	if (txTail >= RBUF_LEN)
		txTail = 0;

	if (txTail != txHead)
		startTransmit();
}

size_t packetSize = sizeof(motor_data_t) + 1;
void serialRxTask(void *arg) {
	motorDataQueueHandle = osMessageQueueNew(8, sizeof(motor_data_t),
			&motorDataQueueAttributes);
	HAL_UART_RegisterCallback(&huart1, HAL_UART_RX_COMPLETE_CB_ID,
			&rxcbComplete);
	HAL_UART_RegisterCallback(&huart1, HAL_UART_TX_COMPLETE_CB_ID,
			&txcbComplete);
	startRec();

	printf("Serial RX starting\r\n");
	char buf[256];
	size_t charsReceived = 0;
	TickType_t lastRec = xTaskGetTickCount();
	while (1) {
		TickType_t now = xTaskGetTickCount();
		if (hasChars()) {
			char c = getChar();
			if (!charsReceived) {
				if (c == 0xAA) {
					charsReceived++;
				}
			} else {
				buf[charsReceived++] = c;
				if (charsReceived >= packetSize) {
					motor_data_t data;
					memcpy((uint8_t*)&data, (uint8_t*)&buf[1], sizeof(data));
					printf("fwd: %05d strafe: %05d turn: %05d\r\n",
							(int) data.fwd, (int) data.strafe, (int) data.turn);
//					for (int i = 1; i < charsReceived; i++) {
//						printf("c: %c\r\n", buf[i]);
//					}
					xQueueSend(motorDataQueueHandle, &data, 10);
					charsReceived = 0;
					lastRec = now;
					connected = 1;
				}
			}
		}
		if (connected && ((now - lastRec) >= 500)) {
			connected = 0;
			motor_data_t data = { .fwd = 0, .strafe = 0, .turn = 0 };
			printf("Timeout!\r\n");
			xQueueSend(motorDataQueueHandle, &data, 10);
		}
	}
}
