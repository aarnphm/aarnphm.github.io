/*
 * Generated with the FPGA Interface C API Generator 18.0.0
 * for NI-RIO 18.0.0 or later.
 */

#ifndef __NiFpga_MyRio1900Fpga60_h__
#define __NiFpga_MyRio1900Fpga60_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1800
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_MyRio1900Fpga60_Bitfile;
 */
#define NiFpga_MyRio1900Fpga60_Bitfile "NiFpga_MyRio1900Fpga60.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_MyRio1900Fpga60_Signature = "056C039B0BCF2FB59C9E5EBD23A16EEF";

typedef enum
{
   NiFpga_MyRio1900Fpga60_IndicatorBool_AOSYSSTAT = 0x1807E,
   NiFpga_MyRio1900Fpga60_IndicatorBool_SYSACCRDY = 0x1824E,
   NiFpga_MyRio1900Fpga60_IndicatorBool_SYSAIRDY = 0x18252,
   NiFpga_MyRio1900Fpga60_IndicatorBool_SYSRDY = 0x18256,
} NiFpga_MyRio1900Fpga60_IndicatorBool;

typedef enum
{
   NiFpga_MyRio1900Fpga60_IndicatorU8_DIBTN = 0x18082,
   NiFpga_MyRio1900Fpga60_IndicatorU8_DIOA_158IN = 0x180AE,
   NiFpga_MyRio1900Fpga60_IndicatorU8_DIOA_70IN = 0x180CA,
   NiFpga_MyRio1900Fpga60_IndicatorU8_DIOB_158IN = 0x180E2,
   NiFpga_MyRio1900Fpga60_IndicatorU8_DIOB_70IN = 0x180EE,
   NiFpga_MyRio1900Fpga60_IndicatorU8_DIOC_70IN = 0x180FA,
   NiFpga_MyRio1900Fpga60_IndicatorU8_ENCASTAT = 0x1810E,
   NiFpga_MyRio1900Fpga60_IndicatorU8_ENCBSTAT = 0x1811A,
   NiFpga_MyRio1900Fpga60_IndicatorU8_ENCC_0STAT = 0x18126,
   NiFpga_MyRio1900Fpga60_IndicatorU8_ENCC_1STAT = 0x18132,
   NiFpga_MyRio1900Fpga60_IndicatorU8_I2CADATI = 0x18146,
   NiFpga_MyRio1900Fpga60_IndicatorU8_I2CASTAT = 0x18152,
   NiFpga_MyRio1900Fpga60_IndicatorU8_I2CBDATI = 0x18166,
   NiFpga_MyRio1900Fpga60_IndicatorU8_I2CBSTAT = 0x18172,
   NiFpga_MyRio1900Fpga60_IndicatorU8_SPIASTAT = 0x1822E,
   NiFpga_MyRio1900Fpga60_IndicatorU8_SPIBSTAT = 0x1824A,
} NiFpga_MyRio1900Fpga60_IndicatorU8;

typedef enum
{
   NiFpga_MyRio1900Fpga60_IndicatorU16_ACCXVAL = 0x18002,
   NiFpga_MyRio1900Fpga60_IndicatorU16_ACCYVAL = 0x18006,
   NiFpga_MyRio1900Fpga60_IndicatorU16_ACCZVAL = 0x1800A,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIA_0VAL = 0x1801E,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIA_1VAL = 0x1802E,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIA_2VAL = 0x18032,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIA_3VAL = 0x18036,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIAudioIn_LVAL = 0x1803A,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIAudioIn_RVAL = 0x1803E,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIB_0VAL = 0x18042,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIB_1VAL = 0x18046,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIB_2VAL = 0x1804A,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIB_3VAL = 0x1804E,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIC_0VAL = 0x18052,
   NiFpga_MyRio1900Fpga60_IndicatorU16_AIC_1VAL = 0x18056,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMA_0CNTR = 0x1817E,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMA_1CNTR = 0x18192,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMA_2CNTR = 0x181A6,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMB_0CNTR = 0x181BA,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMB_1CNTR = 0x181CE,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMB_2CNTR = 0x181E2,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMC_0CNTR = 0x181F6,
   NiFpga_MyRio1900Fpga60_IndicatorU16_PWMC_1CNTR = 0x1820A,
   NiFpga_MyRio1900Fpga60_IndicatorU16_SPIADATI = 0x1821E,
   NiFpga_MyRio1900Fpga60_IndicatorU16_SPIBDATI = 0x1823A,
} NiFpga_MyRio1900Fpga60_IndicatorU16;

typedef enum
{
   NiFpga_MyRio1900Fpga60_IndicatorU32_ENCACNTR = 0x18108,
   NiFpga_MyRio1900Fpga60_IndicatorU32_ENCBCNTR = 0x18114,
   NiFpga_MyRio1900Fpga60_IndicatorU32_ENCC_0CNTR = 0x18120,
   NiFpga_MyRio1900Fpga60_IndicatorU32_ENCC_1CNTR = 0x1812C,
   NiFpga_MyRio1900Fpga60_IndicatorU32_IRQTIMERREAD = 0x18264,
} NiFpga_MyRio1900Fpga60_IndicatorU32;

typedef enum
{
   NiFpga_MyRio1900Fpga60_ControlBool_AOSYSGO = 0x1807A,
   NiFpga_MyRio1900Fpga60_ControlBool_I2CAGO = 0x1814E,
   NiFpga_MyRio1900Fpga60_ControlBool_I2CBGO = 0x1816E,
   NiFpga_MyRio1900Fpga60_ControlBool_IRQDI_BTNENA = 0x1808A,
   NiFpga_MyRio1900Fpga60_ControlBool_IRQDI_BTNFALL = 0x1808E,
   NiFpga_MyRio1900Fpga60_ControlBool_IRQDI_BTNRISE = 0x18096,
   NiFpga_MyRio1900Fpga60_ControlBool_IRQTIMERSETTIME = 0x1826A,
   NiFpga_MyRio1900Fpga60_ControlBool_SPIAGO = 0x18226,
   NiFpga_MyRio1900Fpga60_ControlBool_SPIBGO = 0x18242,
} NiFpga_MyRio1900Fpga60_ControlBool;

typedef enum
{
   NiFpga_MyRio1900Fpga60_ControlU8_DIOA_158DIR = 0x180AA,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOA_158OUT = 0x180B2,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOA_70DIR = 0x180C6,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOA_70OUT = 0x180DA,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOB_158DIR = 0x180DE,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOB_158OUT = 0x180E6,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOB_70DIR = 0x180EA,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOB_70OUT = 0x180F2,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOC_70DIR = 0x180F6,
   NiFpga_MyRio1900Fpga60_ControlU8_DIOC_70OUT = 0x180FE,
   NiFpga_MyRio1900Fpga60_ControlU8_DOLED30 = 0x18102,
   NiFpga_MyRio1900Fpga60_ControlU8_ENCACNFG = 0x18106,
   NiFpga_MyRio1900Fpga60_ControlU8_ENCBCNFG = 0x18112,
   NiFpga_MyRio1900Fpga60_ControlU8_ENCC_0CNFG = 0x1811E,
   NiFpga_MyRio1900Fpga60_ControlU8_ENCC_1CNFG = 0x1812A,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CAADDR = 0x18136,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CACNFG = 0x1813A,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CACNTL = 0x1813E,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CACNTR = 0x18142,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CADATO = 0x1814A,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CBADDR = 0x18156,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CBCNFG = 0x1815A,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CBCNTL = 0x1815E,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CBCNTR = 0x18162,
   NiFpga_MyRio1900Fpga60_ControlU8_I2CBDATO = 0x1816A,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQAI_A_0NO = 0x18016,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQAI_A_1NO = 0x18026,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQAI_A_30CNFG = 0x1800E,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_0NO = 0x1809E,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_1NO = 0x180A6,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_2NO = 0x180BA,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_3NO = 0x180C2,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_70ENA = 0x180CE,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_70FALL = 0x180D2,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDIO_A_70RISE = 0x180D6,
   NiFpga_MyRio1900Fpga60_ControlU8_IRQDI_BTNNO = 0x18092,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMA_0CNFG = 0x1817A,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMA_0CS = 0x18182,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMA_1CNFG = 0x1818E,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMA_1CS = 0x18196,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMA_2CNFG = 0x181A2,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMA_2CS = 0x181AA,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMB_0CNFG = 0x181B6,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMB_0CS = 0x181BE,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMB_1CNFG = 0x181CA,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMB_1CS = 0x181D2,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMB_2CNFG = 0x181DE,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMB_2CS = 0x181E6,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMC_0CNFG = 0x181F2,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMC_0CS = 0x181FA,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMC_1CNFG = 0x18206,
   NiFpga_MyRio1900Fpga60_ControlU8_PWMC_1CS = 0x1820E,
   NiFpga_MyRio1900Fpga60_ControlU8_SPIAIRQNO = 0x1822A,
   NiFpga_MyRio1900Fpga60_ControlU8_SPIBIRQNO = 0x18246,
   NiFpga_MyRio1900Fpga60_ControlU8_SYSSELECTA = 0x1825A,
   NiFpga_MyRio1900Fpga60_ControlU8_SYSSELECTB = 0x1825E,
   NiFpga_MyRio1900Fpga60_ControlU8_SYSSELECTC = 0x18262,
} NiFpga_MyRio1900Fpga60_ControlU8;

typedef enum
{
   NiFpga_MyRio1900Fpga60_ControlU16_AOA_0VAL = 0x1805A,
   NiFpga_MyRio1900Fpga60_ControlU16_AOA_1VAL = 0x1805E,
   NiFpga_MyRio1900Fpga60_ControlU16_AOAudioOut_LVAL = 0x18062,
   NiFpga_MyRio1900Fpga60_ControlU16_AOAudioOut_RVAL = 0x18066,
   NiFpga_MyRio1900Fpga60_ControlU16_AOB_0VAL = 0x1806A,
   NiFpga_MyRio1900Fpga60_ControlU16_AOB_1VAL = 0x1806E,
   NiFpga_MyRio1900Fpga60_ControlU16_AOC_0VAL = 0x18072,
   NiFpga_MyRio1900Fpga60_ControlU16_AOC_1VAL = 0x18076,
   NiFpga_MyRio1900Fpga60_ControlU16_IRQAI_A_0HYSTERESIS = 0x18012,
   NiFpga_MyRio1900Fpga60_ControlU16_IRQAI_A_0THRESHOLD = 0x1801A,
   NiFpga_MyRio1900Fpga60_ControlU16_IRQAI_A_1HYSTERESIS = 0x18022,
   NiFpga_MyRio1900Fpga60_ControlU16_IRQAI_A_1THRESHOLD = 0x1802A,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMA_0CMP = 0x18176,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMA_0MAX = 0x18186,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMA_1CMP = 0x1818A,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMA_1MAX = 0x1819A,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMA_2CMP = 0x1819E,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMA_2MAX = 0x181AE,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMB_0CMP = 0x181B2,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMB_0MAX = 0x181C2,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMB_1CMP = 0x181C6,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMB_1MAX = 0x181D6,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMB_2CMP = 0x181DA,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMB_2MAX = 0x181EA,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMC_0CMP = 0x181EE,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMC_0MAX = 0x181FE,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMC_1CMP = 0x18202,
   NiFpga_MyRio1900Fpga60_ControlU16_PWMC_1MAX = 0x18212,
   NiFpga_MyRio1900Fpga60_ControlU16_SPIACNFG = 0x18216,
   NiFpga_MyRio1900Fpga60_ControlU16_SPIACNT = 0x1821A,
   NiFpga_MyRio1900Fpga60_ControlU16_SPIADATO = 0x18222,
   NiFpga_MyRio1900Fpga60_ControlU16_SPIBCNFG = 0x18232,
   NiFpga_MyRio1900Fpga60_ControlU16_SPIBCNT = 0x18236,
   NiFpga_MyRio1900Fpga60_ControlU16_SPIBDATO = 0x1823E,
} NiFpga_MyRio1900Fpga60_ControlU16;

typedef enum
{
   NiFpga_MyRio1900Fpga60_ControlU32_IRQDIO_A_0CNT = 0x18098,
   NiFpga_MyRio1900Fpga60_ControlU32_IRQDIO_A_1CNT = 0x180A0,
   NiFpga_MyRio1900Fpga60_ControlU32_IRQDIO_A_2CNT = 0x180B4,
   NiFpga_MyRio1900Fpga60_ControlU32_IRQDIO_A_3CNT = 0x180BC,
   NiFpga_MyRio1900Fpga60_ControlU32_IRQDI_BTNCNT = 0x18084,
   NiFpga_MyRio1900Fpga60_ControlU32_IRQTIMERWRITE = 0x1826C,
} NiFpga_MyRio1900Fpga60_ControlU32;

#endif
