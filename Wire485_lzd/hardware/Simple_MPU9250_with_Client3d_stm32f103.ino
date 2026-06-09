/* ============================================
  Simple_MPU6050 device library code is placed under the MIT license
  Copyright (c) 2021 Homer Creutz

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
  ===============================================
*/

/*
 *  Use with any MPU: MPU6050, MPU6500, MPU9150, MPU9155, MPU9250 
 *  Attach the MPU to the I2C bus
 *  Power MPU According to specs of the breakout board. Generic Breakout Version Powers with 5V and has an onboard Voltage regulator.
 *  run the Sketch
 */


#define QUAT_ANIMATION// Uncomment this line to output data in the correct format for ZaneL's Node.js Quaternion animation tool: https://github.com/ZaneL/quaternion_sensor_3d_nodejs

#include "Simple_MPU6050.h"
#define MPU6050_DEFAULT_ADDRESS     0x68 // address pin low (GND), default for InvenSense evaluation board

char buf[1000];

Simple_MPU6050 mpu;
//d:\GSHdocument\imu\libraries\Simple_MPU6050\Simple_MPU6050.h
#define ID_130

#ifdef ID_128
char id_self = 128;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS   3140,    4052,    9424,     -33,     -16,      -1
#endif

#ifdef ID_129
char id_self = 129;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS  -5134,   -4910,    8430,     -59,      83,      10
#endif

#ifdef ID_130
char id_self = 130;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS  -6130,   -4532,    7734,     323,      99,      24
#endif

#ifdef ID_131
char id_self = 131;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS  -7452,   -5244,    8438,     174,     -60,      15
#endif

#ifdef ID_132
char id_self = 132;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS  -6658,    4914,    8398,    -139,      89,     -73
#endif

#ifdef ID_133
char id_self = 133;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS   3438,   -5304,    9570,      71,      56,     -67
#endif

#ifdef ID_134
char id_self = 134;
//              X Accel  Y Accel  Z Accel   X Gyro   Y Gyro   Z Gyro
#define OFFSETS   3158,    4044,    9430,     -30,     -19,      -3
#endif

char id_read = 0;
int mode = 0; //recv mode
Quaternion q;
char ret;
char id_read_0;
char id_read_1;
char id_read_2;

// char ret[4] = {0, 0, 0, 0};
// int cnt = 0;


int16_t my_gyroo[3];
int16_t my_accel[6];
bool fifo_ok = false;
int msg_count = 10;

// #define AUTO_SEND
//***************************************************************************************
//******************              Callback Funciton                **********************
//***************************************************************************************

// See mpu.on_FIFO(print_Values); in the Setup Loop
void Print_Values (int16_t *gyro, int16_t *accel, int32_t *quat) {

  VectorFloat gravity;
  VectorInt16 acc_raw(accel[0],accel[1],accel[2]);
  VectorInt16 acc_real,acc_inworld;
  float ypr[3] = { 0, 0, 0 };
  float xyz[3] = { 0, 0, 0 };
  my_gyroo[0] = gyro[0];
  my_gyroo[1] = gyro[1];
  my_gyroo[2] = gyro[2];
  // my_accel[0] = accel[0];
  // my_accel[1] = accel[1];
  // my_accel[2] = accel[2];
  // my_accel[0] = acc_raw.x;
  // my_accel[1] = acc_raw.y;
  // my_accel[2] = acc_raw.z;
  // my_accel[0] = acc_raw.x;
  // my_accel[1] = acc_raw.y;
  // my_accel[2] = acc_raw.z;
  fifo_ok = true;

  mpu.GetQuaternion(&q, quat);
  mpu.GetGravity(&gravity, &q);
  mpu.GetYawPitchRoll(ypr, &q, &gravity);
  mpu.ConvertToDegrees(ypr, xyz);
  mpu.GetLinearAccel(&acc_real, &acc_raw,&gravity);
  mpu.GetLinearAccelInWorld(&acc_inworld, &acc_real, &q);
  my_accel[0] = acc_raw.x;
  my_accel[1] = acc_raw.y;
  my_accel[2] = acc_raw.z;
  my_accel[3] = acc_real.x;
  my_accel[4] = acc_real.y;
  my_accel[5] = acc_real.z;

#ifndef QUAT_ANIMATION
  // Serial.print(F("quat"));Serial.print(q.w,6);Serial.print(F(","));Serial.print(q.x,6);Serial.print(F(","));Serial.print(q.y,6);Serial.print(F(","));Serial.print(q.z,6);Serial.print(F("\n"));
  // Serial.print(F("acce"));Serial.print(accel[0]); Serial.print(F(","));Serial.print(accel[1]); Serial.print(F(","));Serial.print(accel[2]); Serial.print(F("\n"));
  Serial.print(F("gravity: "));Serial.print(gravity.x); Serial.print(F(","));Serial.print(gravity.y); Serial.print(F(","));Serial.print(gravity.z); Serial.print(F("\n"));
  Serial.print(F("acc_raw: "));Serial.print(acc_raw.x); Serial.print(F(","));Serial.print(acc_raw.y); Serial.print(F(","));Serial.print(acc_raw.z); Serial.print(F("\n"));
  Serial.print(F("acc_real: "));Serial.print(acc_real.x); Serial.print(F(","));Serial.print(acc_real.y); Serial.print(F(","));Serial.print(acc_real.z); Serial.print(F("\n"));
  Serial.print(F("acc_inworld: "));Serial.print(acc_inworld.x); Serial.print(F(","));Serial.print(acc_inworld.y); Serial.print(F(","));Serial.print(acc_inworld.z); Serial.print(F("\n"));
#else
  //Output the Quaternion data in the format expected by ZaneL's Node.js Quaternion animation tool
  // digitalWrite(PB0, 1);
  // // Serial.print(F("{\"w\":"));
  // // Serial.print(q.w, 4);
  // // Serial.print(F(", \"x\":"));
  // // Serial.print(q.x, 4);
  // // Serial.print(F(", \"y\":"));
  // // Serial.print(q.y, 4);
  // // Serial.print(F(", \"z\":"));
  // // Serial.print(q.z, 4);
  // // Serial.print(F("}"));
  // // Serial.write(id_self+1);
  // // Serial.write('\n');
  // // Serial.flush ();
  // memset(buf,'a',500);
  // buf[499] = '\n';
  // Serial.write(buf,500);
  // Serial.flush ();
  //   // Serial.write('\n');
  // digitalWrite(PB0, 0);

  // if(msg_count>0)
  // {
  //   digitalWrite(PB0, 1);
  //   // memcpy(&buf[0],&(q.w),4);
  //   // memcpy(&buf[4],&(q.x),4);
  //   // memcpy(&buf[8],&(q.y),4);
  //   // memcpy(&buf[12],&(q.z),4);
  //   // buf[16]=id_self+1;
  //   // buf[17]='\n';
  //   // Serial.write(buf,18);
  //   Serial.write("123456789\n",10);
  //   Serial.flush ();//wait for send finished
  //   digitalWrite(PB0, 0);
  // }
  // msg_count--;

#endif
  // Serial.print(F("Yaw "));   
  // Serial.print(xyz[0]);   Serial.print(F(",   "));
  // Serial.print(F("Pitch ")); 
  // Serial.print(xyz[1]);   Serial.print(F(",   "));
  // Serial.print(F("Roll "));  
  // Serial.print(xyz[2]);   Serial.print(F(",   "));
  // Serial.print(F("ax "));    
  // Serial.print(accel[0]); Serial.print(F(",   "));
  // Serial.print(F("ay "));    
  // Serial.print(accel[1]); Serial.print(F(",   "));
  // Serial.print(F("az "));    
  // Serial.print(accel[2]); Serial.print(F(",   "));
  // Serial.print(F("gx "));    Serial.print(gyro[0]);  Serial.print(F(",   "));
  // Serial.print(F("gy "));    Serial.print(gyro[1]);  Serial.print(F(",   "));
  // Serial.print(F("gz "));    Serial.print(gyro[2]);  Serial.print(F("\n"));
  // Serial.println();
}

//***************************************************************************************
//******************                Setup and Loop                 **********************
//***************************************************************************************

void setup() {
  // initialize serial communication
  pinMode(PB0, OUTPUT);
  digitalWrite(PB0, 0);
  Serial.setTx(PA9);
  Serial.setRx(PA10);
  Serial.begin(921600);
  while (!Serial); // wait for Leonardo enumeration, others continue immediately
#ifndef QUAT_ANIMATION
  Serial.println(F("Start:"));
#endif
  // Setup the MPU and TwoWire aka Wire library all at once
  mpu.begin(PB7,PB6);  
  mpu.Set_DMP_Output_Rate_Hz(200);          // Set the DMP output rate from 200Hz to 5 Minutes.
  //mpu.Set_DMP_Output_Rate_Seconds(10);   // Set the DMP output rate in Seconds
  //mpu.Set_DMP_Output_Rate_Minutes(5);    // Set the DMP output rate in Minute

#ifdef OFFSETS
#ifndef QUAT_ANIMATION
  Serial.println(F("Using Offsets"));
#endif
  // mpu.SetAddress(MPU6050_DEFAULT_ADDRESS);
  mpu.load_DMP_Image(OFFSETS); // Does it all for you

#else
#ifndef QUAT_ANIMATION
  Serial.println(F(" Since no offsets are defined we are going to calibrate this specific MPU6050,\n"
                   " Start by having the MPU6050 placed stationary on a flat surface to get a proper accelerometer calibration\n"
                   " Place the new offsets on the #define OFFSETS... line at top of program for super quick startup\n\n"
                   " \t\t\t[Press Any Key]"));
#endif
  while (Serial.available() && Serial.read()); // empty buffer
  while (!Serial.available());                 // wait for data
  while (Serial.available() && Serial.read()); // empty buffer again
  // mpu.SetAddress(MPU6050_DEFAULT_ADDRESS);
  digitalWrite(PB0, 1);
  mpu.CalibrateMPU();
  mpu.load_DMP_Image();// Does it all for you with Calibration
  Serial.flush();
  digitalWrite(PB0, 0);
#endif
  mpu.WriteByte(0x1A, 6);
  mpu.WriteByte(0x1D, 6);
  // mpu.CalibrateMPU();                      // Calibrates the MPU.
  // mpu.load_DMP_Image();                    // Loads the DMP image into the MPU and finish configuration.
  mpu.on_FIFO(Print_Values);               // Set callback function that is triggered when FIFO Data is retrieved
  // Setup is complete!
  
}

void loop() {
  // static unsigned long FIFO_DelayTimer;
  // if ((millis() - FIFO_DelayTimer) >= (9)) { // 99ms instead of 100ms to start polling the MPU 1ms prior to data arriving.
  //   if( mpu.dmp_read_fifo(false)) FIFO_DelayTimer= millis() ; // false = no interrupt pin attachment required and When data arrives in the FIFO Buffer reset the timer
  // }
  // char ret;
  // char id_read_0;
  // char id_read_1;

  
  while (Serial.available())
  {

    ret = Serial.read();
    // ret[cnt] = Serial.read();
    // cnt++;
    // if (cnt == 4){
    //   cnt = 0;
    // }
    if((ret == 0x0a) && (id_read_0 == id_self) && (id_read_1 == 0xFF) && (id_read_2 == 0xFF))
    // if((ret == 0x0a) && (id_read_0 == id_self) && (id_read_1 == id_self))
    // if((ret[0] == 0xFF) && (ret[1] == 0xFF) && (ret[2]== 0xFF)&& (ret[3]== id_self))
      {
        // delayMicroseconds(100);
        digitalWrite(PB0, 1);
        memcpy(&buf[0],&(q.w),4);
        memcpy(&buf[4],&(q.x),4);
        memcpy(&buf[8],&(q.y),4);
        memcpy(&buf[12],&(q.z),4);
        memcpy(&buf[16],&my_gyroo[0],2);
        memcpy(&buf[18],&my_gyroo[1],2);
        memcpy(&buf[20],&my_gyroo[2],2);
        memcpy(&buf[22],&my_accel[0],2);
        memcpy(&buf[24],&my_accel[1],2);
        memcpy(&buf[26],&my_accel[2],2);
        memcpy(&buf[28],&my_accel[3],2);
        memcpy(&buf[30],&my_accel[4],2);
        memcpy(&buf[32],&my_accel[5],2);


        // memcpy(&buf[12],&my_gyroo[3],4);

        // buf[22]=id_self+1;
        // buf[23]='\n' ;
       
        // buf[28]=id_self+1;
        // buf[29]=id_self+1;
        // buf[30]='\n';

        buf[34]=0xFF;
        buf[35]=0xFF;
        // buf[30]=0xFF;
        buf[36]=id_self+1;
        buf[37]='\n';

        // buf[16]=id_self+1;
        // buf[17]='\n';
    
        // Serial.write(buf,24);
        Serial.write(buf,38);
        // Serial.printf("%d,%d,%d",my_gyroo[0],my_gyroo[1],my_gyroo[2]);
    
        // Serial.write(buf,18);
        Serial.flush ();//wait for send finished
        // delay(1);
        digitalWrite(PB0, 0);
        // delayMicroseconds(100);
        // Serial.print(buf);
      }
    id_read_2 = id_read_1;
    id_read_1 = id_read_0;
    id_read_0 = ret;
    // Serial.print(buf);
  }

  //串口助手发16进制ID，再发0A
  // delayMicroseconds(500);

  mpu.dmp_read_fifo(true);
  // mpu.dmp_read_fifo(true);
  // dmp_read_fifo(false) does the following
  // Tests for Data in the FIFO Buffer
  // when it finds data it runs the mpu.on_FIFO(print_Values)
  // the print_Values function which we set run the PrintAllValues Function
  // When data is captured dmp_read_fifo will return true.
  // The print_Values function MUST have the following variables available to attach data
  // void print_Values (int16_t *gyro, int16_t *accel, int32_t *quat, uint32_t *timestamp)
  // Variables:
  // int16_t *gyro for the gyro values to be passed to it (The * tells the function it will be a pointer to the value)
  // int16_t *accel for the accel values to be passed to it
  // int32_t *quat for the quaternion values to be passed to it
  // uint32_t *timestamp which will be the micros()value at the time we retrieved the Newest value from the FIFO Buffer.
}