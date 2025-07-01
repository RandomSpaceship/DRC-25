/*
 * stdout.c
 *
 *  Created on: Jul 1, 2025
 *      Author: jcnic
 */

#include <stdio.h>
#include "usart.h"

#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif

PUTCHAR_PROTOTYPE
{
	transmitPrintf(ch);
  return ch;
}
