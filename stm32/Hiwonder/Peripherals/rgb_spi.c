
#include "rgb_spi.h"
#include "global.h"
#include "lwmem_porting.h"

static uint8_t pixelBuffer[Pixel_S1_NUM+1][24]; //灯珠
//uint8_t *pixelBuffer;

static uint32_t  rgb_pixel[Pixel_S1_NUM];

void rgb_SendArray(void){
	HAL_SPI_Transmit_DMA(&hspi1, (uint8_t*)pixelBuffer, (Pixel_S1_NUM+1)*24);
}

void WS2812b_Configuration(void)
{
    memset(pixelBuffer , CODE0 , Pixel_S1_NUM*24);
    memset(pixelBuffer[Pixel_S1_NUM] , 0x00 , 24);
}

void set_id_rgb_color(uint8_t id , uint8_t* rgb)
{
    if(id >= Pixel_S1_NUM) //RGB灯编号为 0 、 1
    {
        return;
    }
    int i = 0;
    for(i=0;i<=7;i++){
		pixelBuffer[id][i]= ( (rgb[0] & (1 << (7 -i)) )? (CODE1):CODE0 );
	}
    for(i=8;i<=15;i++){
		pixelBuffer[id][i]= ( (rgb[1] & (1 << (15-i)) )? (CODE1):CODE0 );
	}
    for(i=16;i<=23;i++){
		pixelBuffer[id][i]= ( (rgb[2] & (1 << (23-i)) )? (CODE1):CODE0 );
	}
    rgb_SendArray();
}

void set_rgb_color(uint8_t* rgb)
{
    int i = 0;
    for(i=0;i<=7;i++){
		pixelBuffer[0][i]= ( (rgb[0] & (1 << (7 -i)) )? (CODE1):CODE0 );
	}
    for(i=8;i<=15;i++){
		pixelBuffer[0][i]= ( (rgb[1] & (1 << (15-i)) )? (CODE1):CODE0 );
	}
    for(i=16;i<=23;i++){
		pixelBuffer[0][i]= ( (rgb[2] & (1 << (23-i)) )? (CODE1):CODE0 );
	}
    for(i=0;i<=7;i++){
		pixelBuffer[1][i]= ( (rgb[3] & (1 << (7 -i)) )? (CODE1):CODE0 );
	}
    for(i=8;i<=15;i++){
		pixelBuffer[1][i]= ( (rgb[4] & (1 << (15-i)) )? (CODE1):CODE0 );
	}
    for(i=16;i<=23;i++){
		pixelBuffer[1][i]= ( (rgb[5] & (1 << (23-i)) )? (CODE1):CODE0 );
	}
    rgb_SendArray();
}
