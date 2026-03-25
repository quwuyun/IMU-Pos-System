import os
import threading
import time
import serial
import struct
import mujoco
import mujoco.viewer
import numpy as np
from serial import SerialException
from scipy.spatial.transform import Rotation


# 全局变量定义
Pelvis = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
L_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
L_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
R_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
R_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
g = 9.8


def quaternion_to_matrix(quaternion):
    # 创建四元数对象(x, y, z, w)
    r = Rotation.from_quat(quaternion)
    # 将四元数转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def matrix_to_quaternion(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    return quaternion

def quaternion_to_euler(quaternion):
    r = Rotation.from_quat(quaternion)
    euler = r.as_euler('XYZ', degrees=True)
    return euler

def euler_to_quaternion(euler):
    r = Rotation.from_euler('XYZ', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

# 串口配置 Serial Port Configuration
class SerialConfig:
    # 串口号
    portName = 'COM5' # /dev/ttyUSB0 or COM5

    # 波特率
    baud = 2000000

    # Can波特率
    canBaud = 1000000


# 设备实例 Device instance
class DeviceModel:
    # region 属性 attribute
    # 设备名称 deviceName
    deviceName = "imu"

    # 设备数据字典 Device Data Dictionary
    deviceData = {}

    # 设备是否开启
    isOpen = False

    # 串口 Serial port
    serialPort = None

    # 串口配置 Serial Port Configuration
    serialConfig = SerialConfig()

    # 临时数组 Temporary array
    TempBytes = []

    # 起始寄存器 Start register
    statReg = None

    # 模式(默认AT指令模式)  AT mode
    isAT = True

    # can id及数据模式 Can ID and data mode
    canid = []
    # endregion

    def __init__(self, deviceName, portName, baud, canBaud, callback_method):
        print("初始化设备模型")
        # 设备名称（自定义） Device Name
        self.deviceName = deviceName
        # 串口号 Serial port number
        self.serialConfig.portName = portName
        # 串口波特率 baud
        self.serialConfig.baud = baud
        # 串口CAN波特率 Can baud
        self.serialConfig.canBaud = canBaud
        self.deviceData = {}
        self.callback_method = callback_method

        self.rot_pelvis = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        self.rot_r_hip = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.rot_r_shank = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        self.rot_l_hip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.rot_l_shank = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        self.imu_num = 5
        self.imu_data_dict = {}
        for i in range(1, self.imu_num + 1):
            self.imu_data_dict[i - 1] = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.A = []
        self.A_G = []
        self.G = []
        self.E = []
        self.Q = []
        self.imu_data = []

        self.model_file = os.path.join(os.path.dirname(__file__), 'model', 'walk_new_quat_body0901.xml')
        self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.r_thigh_ori_init = []
        self.r_shank_ori_init = []
        self.l_thigh_ori_init = []
        self.l_shank_ori_init = []
        self.pelvis_ori_init = []

        self.count = 0
        self.det_t = 0.01
        self.last_qvel = np.zeros(30)
        self.LIST = np.zeros(100)
        self.output = []
        self.position_dict = np.zeros(30)
        self.posture_dict = np.zeros(90)
        self.FINAL_data = np.zeros(159)
        self.FINAL_LIST = np.zeros(159)
        self.FINAL_output = np.zeros(159)

        self.Pelvis = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.L_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.L_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.R_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.R_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # region 获取设备数据 Obtain device data
    # 设置设备数据 Set device data
    def set(self, key, value):
        # 将设备数据存到键值 Saving device data to key values
        self.deviceData[key] = value

    # 获得设备数据 Obtain device data
    def get(self, key):
        # 从键值中获取数据，没有则返回None Obtaining data from key values
        if key in self.deviceData:
            return self.deviceData[key]
        else:
            return None

    # 删除设备数据 Delete device data
    def remove(self, key):
        # 删除设备键值
        del self.deviceData[key]
    # endregion

    # 打开设备 open Device
    def openDevice(self):
        # 先关闭端口 Turn off the device first
        self.closeDevice()
        try:
            self.serialPort = serial.Serial(self.serialConfig.portName, self.serialConfig.baud, timeout=0.5)
            self.isOpen = True
            print("{}已打开".format(self.serialConfig.portName))

            # 设置USB-CAN模块CAN波特率 Set USB-CAN module CAN baud rate
            self.sendData("AT+CG\r\n".encode())
            time.sleep(0.1)
            self.sendData("AT+CAN_BAUD={}\r\n".format(self.serialConfig.canBaud).encode())
            print("设置CAN波特率为{}".format(self.serialConfig.canBaud))
            time.sleep(0.1)
            self.sendData("AT+AT\r\n".encode())

            # 开启一个线程持续监听串口数据 Start a thread to continuously listen to serial port data
            t = threading.Thread(target=self.readDataTh, args=("Data-Received-Thread", 10,))
            t.start()
            print("设备打开成功")
        except SerialException:
            print("打开" + self.serialConfig.portName + "失败")

    # 监听串口数据线程 Listening to serial data threads
    def readDataTh(self, threadName, delay):
        print("启动" + threadName)
        while True:
            # 如果串口打开了
            if self.isOpen:
                try:
                    tLen = self.serialPort.inWaiting()
                    if tLen > 0:
                        data = self.serialPort.read(tLen)
                        self.onDataReceived(data)
                except Exception as ex:
                    print(ex)
            else:
                time.sleep(0.1)
                print("串口未打开")
                break

    # 关闭设备  close Device
    def closeDevice(self):
        if self.serialPort is not None:
            self.serialPort.close()
            print("端口关闭了")
        self.isOpen = False
        print("设备关闭了")

    # region 数据解析 data analysis
    # 串口数据处理  Serial port data processing
    def onDataReceived(self, data):
        tempdata = bytes.fromhex(data.hex())
        # AT指令模式  AT mode
        if self.isAT:
            for val in tempdata:
                self.TempBytes.append(val)
                if len(self.TempBytes) > 7:
                    # AT
                    if not ((self.TempBytes[0] == 0x41) and (self.TempBytes[1] == 0x54)):
                        del self.TempBytes[0]
                        continue
                    tLen = len(self.TempBytes)
                    # 长度验证 Length verification
                    if tLen == self.TempBytes[6] + 9:
                        if not (self.TempBytes[15] == 0x0D and self.TempBytes[16] == 0x0A):
                            del self.TempBytes[0]
                            continue
                        # 协议头解析 Protocol header parsing
                        self.processProtocol(self.TempBytes[2:6])
                        # 数据解析 Data parsing
                        self.processData(self.TempBytes[7:15])
                        self.TempBytes.clear()
        # 透传模式 Transparent mode
        else:
            for val in tempdata:
                self.TempBytes.append(val)
                if self.TempBytes[0] != 0x55:
                    del self.TempBytes[0]
                    continue
                tLen = len(self.TempBytes)
                if tLen == 8:
                    self.processData(self.TempBytes)
                    self.set("canmode_1", "透传")
                    self.set("canmode_2", " ")
                    self.TempBytes.clear()

    # CANID解析
    def processProtocol(self, Bytes):
        self.canid = Bytes
        bytes_data = bytearray(Bytes)
        # 拿到二进制序列 Get the binary sequence
        binary_data = bin(struct.unpack("!I", bytes_data)[0])[2:].zfill(32)
        if binary_data[30] == '0':
            self.set("canmode_1", "数据帧")
        else:
            self.set("canmode_1", "远程帧")
        if binary_data[29] == '0':
            self.set("canmode_2", "标准")
            can_id = int(binary_data[:11], 2)
            self.set("CanID", can_id)
        else:
            self.set("canmode_2", "拓展")
            can_id = int(binary_data[:29], 2)
            self.set("CanID", can_id)

    # 数据解析 data analysis
    def processData(self, Bytes):
        if Bytes[1] == 0x50:
            pass
        # 加速度 Acceleration
        elif Bytes[1] == 0x51:
            Ax = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 16 * g
            Ay = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 16 * g
            Az = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 16 * g
            A_Gz = Az - g
            self.set("AccX", round(Ax, 3))
            self.set("AccY", round(Ay, 3))
            self.set("AccZ", round(Az, 3))
            self.callback_method(self)
            self.A = [Ax, Ay, Az]
            # print("A:{}".format(self.A))
            self.A_G = [Ax, Ay, A_Gz]
            # print("A_G:{}".format(self.A_G))
        # 角速度 Angular velocity
        elif Bytes[1] == 0x52:
            Gx = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 2000
            Gy = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 2000
            Gz = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 2000
            self.set("AsX", round(Gx, 3))
            self.set("AsY", round(Gy, 3))
            self.set("AsZ", round(Gz, 3))
            self.G = [Gx, Gy, Gz]
            # print("G:{}".format(self.G))
        # 角度 Angle
        elif Bytes[1] == 0x53:
            AngX = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 180
            AngY = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 180
            AngZ = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 180
            self.set("AngX", round(AngX, 3))
            self.set("AngY", round(AngY, 3))
            self.set("AngZ", round(AngZ, 3))
            self.E = [AngX, AngY, AngZ]
            # print("CanID {}: E:{}".format(self.get("CanID"), format(self.E))
            self.Q = euler_to_quaternion(self.E).tolist()
            # print("Q:{}".format(self.Q))
            self.imu_data = [self.Q, self.G, self.A, self.A_G]
            # print("imu_data:{}".format(self.imu_data))
            i = self.get("CanID") - 79
            self.imu_data_dict[i - 1] = self.imu_data
            # print("CanID {}: {}".format(self.get("CanID"), self.imu_data_dict[i - 1]))
            self.count = self.count + 1
        # 磁场 Magnetic field
        elif Bytes[1] == 0x54:
            Hx = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) * 8.333 / 1000
            Hy = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) * 8.333 / 1000
            Hz = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) * 8.333 / 1000
            self.set("HX", round(Hx, 3))
            self.set("HY", round(Hy, 3))
            self.set("HZ", round(Hz, 3))
        # 读取回传 Read callback
        elif Bytes[1] == 0x5F:
            value = self.getSignInt16(Bytes[3] << 8 | Bytes[2])
            self.set(str(self.statReg), value)
            print("读取数据返回  reg:{}   value:{} ".format(self.statReg, value))
        else:
            pass


    # 获得int16有符号数 Obtain int16 signed number
    def getSignInt16(self, num):
        if num >= pow(2, 15):
            num -= pow(2, 16)
        return num
    # endregion

    # 发送串口数据 Sending serial port data
    def sendData(self, data):
        try:
            self.serialPort.write(data)
        except Exception as ex:
            print(ex)

    # 读取寄存器 read register
    def readReg(self, regAddr):
        # 从指令中获取起始寄存器 （处理回传数据需要用到） Get start register from instruction
        self.statReg = regAddr
        # 封装读取指令并向串口发送数据 Encapsulate read instructions and send data to the serial port
        self.sendData(self.get_readBytes(regAddr))

    # 写入寄存器 Write Register
    def writeReg(self, regAddr, sValue):
        # 解锁 unlock
        self.unlock()
        # 延迟100ms Delay 100ms
        time.sleep(0.1)
        # 封装写入指令并向串口发送数据
        self.sendData(self.get_writeBytes(regAddr, sValue))
        # 延迟100ms Delay 100ms
        time.sleep(0.1)
        # 保存 save
        self.save()

    # 读取指令封装 Read instruction encapsulation
    def get_readBytes(self, regAddr):
        # 初始化
        tempBytes = [None] * 14
        # 设备modbus地址
        tempBytes[0] = 0x41
        # 读取功能码
        tempBytes[1] = 0x54
        # 寄存器高8位
        tempBytes[2] = self.canid[0]
        # 寄存器低8位
        tempBytes[3] = self.canid[1]
        # 读取寄存器个数高8位
        tempBytes[4] = self.canid[2]
        # 读取寄存器个数低8位
        tempBytes[5] = self.canid[3]
        tempBytes[6] = 5
        tempBytes[7] = 0xFF
        tempBytes[8] = 0xAA
        tempBytes[9] = 0x27
        tempBytes[10] = regAddr & 0xff
        tempBytes[11] = regAddr >> 8
        tempBytes[12] = 0x0D
        tempBytes[13] = 0x0A
        return tempBytes

    # 写入指令封装 Write instruction encapsulation
    def get_writeBytes(self, regAddr, rValue):
        # 初始化
        tempBytes = [None] * 14
        # 设备modbus地址
        tempBytes[0] = 0x41
        # 读取功能码
        tempBytes[1] = 0x54
        # 寄存器高8位
        tempBytes[2] = self.canid[0]
        # 寄存器低8位
        tempBytes[3] = self.canid[1]
        # 读取寄存器个数高8位
        tempBytes[4] = self.canid[2]
        # 读取寄存器个数低8位
        tempBytes[5] = self.canid[3]
        tempBytes[6] = 5
        tempBytes[7] = 0xFF
        tempBytes[8] = 0xAA
        tempBytes[9] = regAddr
        tempBytes[10] = rValue & 0xff
        tempBytes[11] = rValue >> 8
        tempBytes[12] = 0x0D
        tempBytes[13] = 0x0A
        return tempBytes

    # 解锁
    def unlock(self):
        cmd = self.get_writeBytes(0x69, 0xb588)
        self.sendData(cmd)

    # 保存
    def save(self):
        cmd = self.get_writeBytes(0x00, 0x0000)
        self.sendData(cmd)

    # 设置AT模式 Set AT mode
    def setAT(self):
        cmd = "AT+AT\r\n"
        self.sendData(cmd.encode())
        self.isAT = True

    # 设置透传模式 Set Transparent mode
    def setET(self):
        cmd = "AT+ET\r\n"
        self.sendData(cmd.encode())
        self.isAT = False

    def start_visualization(self):
        """启动MuJoCo可视化窗口"""
        if self.model and self.data:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def update_visualization(self):
        """更新可视化显示"""
        if self.viewer:
            self.viewer.sync()


def imu_data_solving(self):
    if self.count < 100:   # 初始化
        # imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        imu_data_list = [self.imu_data_dict[i] for i in range(0, 5)]
        self.R_shank, self.R_thigh, self.Pelvis, self.L_thigh, self.L_shank = imu_data_list

        r_thigh_quaternion_init = self.R_thigh[0]
        r_shank_quaternion_init = self.R_shank[0]
        l_thigh_quaternion_init = self.L_thigh[0]
        l_shank_quaternion_init = self.L_shank[0]
        pelvis_quaternion_init = self.Pelvis[0]

        self.r_thigh_ori_init = quaternion_to_matrix(r_thigh_quaternion_init)
        self.r_shank_ori_init = quaternion_to_matrix(r_shank_quaternion_init)
        self.l_thigh_ori_init = quaternion_to_matrix(l_thigh_quaternion_init)
        self.l_shank_ori_init = quaternion_to_matrix(l_shank_quaternion_init)
        self.pelvis_ori_init = quaternion_to_matrix(pelvis_quaternion_init)

    elif self.count == 100:
        print('start')

    else:
        # imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        imu_data_list = [self.imu_data_dict[i] for i in range(0, 5)]
        self.R_shank, self.R_thigh, self.Pelvis, self.L_thigh, self.L_shank = imu_data_list

        # print("imu_data_list:{}".format(imu_data_list))
        # print("L_thigh:{}".format(self.L_thigh))
        # print("L_shank:{}".format(self.L_shank))

        '''骨盆位置'''  # X向前，Y向左，Z向上
        # Position = camera[0]
        # data.qpos[0] = -Position.z
        # data.qpos[1] = -Position.x
        # data.qpos[2] = Position.y
        pelvis_quaternion = self.Pelvis[0]
        pelvis_ori = quaternion_to_matrix(pelvis_quaternion)
        Pelvis_ori = self.pelvis_ori_init.T @ pelvis_ori
        pelvis_quat = matrix_to_quaternion(Pelvis_ori)
        PELVIS_quat = self.rot_pelvis @ [pelvis_quat[0], pelvis_quat[1], pelvis_quat[2]]
        self.data.qpos[3] = pelvis_quat[3]
        self.data.qpos[4] = -PELVIS_quat[0]  # 外摆
        self.data.qpos[5] = -PELVIS_quat[2]
        self.data.qpos[6] = -PELVIS_quat[1]  # 弯曲
        T_pelvis = quaternion_to_matrix([-PELVIS_quat[0], -PELVIS_quat[2], -PELVIS_quat[1], pelvis_quat[3]])

        '''右髋关节角度'''
        r_thigh_quaternion = self.R_thigh[0]
        r_thigh_ori = quaternion_to_matrix(r_thigh_quaternion)
        R_thigh_ori = self.r_thigh_ori_init.T @ r_thigh_ori
        r_thigh_quat = matrix_to_quaternion(R_thigh_ori)
        r_hip_quat = self.rot_r_hip @ [r_thigh_quat[0], r_thigh_quat[1], r_thigh_quat[2]]
        T_r_thigh = quaternion_to_matrix([-r_hip_quat[0], -r_hip_quat[2], -r_hip_quat[1], r_thigh_quat[3]])
        T_r_thigh_pelvis = T_pelvis.T @ T_r_thigh
        r_hip = matrix_to_quaternion(T_r_thigh_pelvis)
        self.data.qpos[7] = r_hip[3]
        self.data.qpos[8] = r_hip[0]  # 外摆
        self.data.qpos[9] = r_hip[1]
        self.data.qpos[10] = r_hip[2]  # 弯曲

        '''右膝关节角度'''
        r_shank_quaternion = self.R_shank[0]
        r_shank_ori = quaternion_to_matrix(r_shank_quaternion)
        R_shank_ori = self.r_shank_ori_init.T @ r_shank_ori
        r_shank_quat = matrix_to_quaternion(R_shank_ori)
        r_knee_quat = self.rot_r_shank @ [r_shank_quat[0], r_shank_quat[1], r_shank_quat[2]]
        T_r_shank = quaternion_to_matrix([-r_knee_quat[0], -r_knee_quat[2], -r_knee_quat[1], r_shank_quat[3]])
        T_r_shank_thigh = T_r_thigh.T @ T_r_shank
        r_knee = matrix_to_quaternion(T_r_shank_thigh)
        self.data.qpos[11] = abs(r_knee[3])
        # self.data.qpos[12] = r_knee[0]
        # self.data.qpos[13] = r_knee[1]
        self.data.qpos[14] = -abs(r_knee[2])  # 弯曲

        '''左髋关节角度'''
        l_thigh_quaternion = self.L_thigh[0]
        l_thigh_ori = quaternion_to_matrix(l_thigh_quaternion)
        L_thigh_ori = self.l_thigh_ori_init.T @ l_thigh_ori
        l_thigh_quat = matrix_to_quaternion(L_thigh_ori)
        l_hip_quat = self.rot_l_hip @ [l_thigh_quat[0], l_thigh_quat[1], l_thigh_quat[2]]
        T_l_thigh = quaternion_to_matrix([-l_hip_quat[0], -l_hip_quat[2], -l_hip_quat[1], l_thigh_quat[3]])
        T_l_thigh_pelvis = T_pelvis.T @ T_l_thigh
        l_hip = matrix_to_quaternion(T_l_thigh_pelvis)
        self.data.qpos[21] = l_hip[3]
        self.data.qpos[22] = l_hip[0]  # 外摆
        self.data.qpos[23] = l_hip[1]
        self.data.qpos[24] = l_hip[2]  # 弯曲

        '''左膝关节角度'''
        l_shank_quaternion = self.L_shank[0]
        l_shank_ori = quaternion_to_matrix(l_shank_quaternion)
        L_shank_ori = self.l_shank_ori_init.T @ l_shank_ori
        l_shank_quat = matrix_to_quaternion(L_shank_ori)
        l_knee_quat = self.rot_l_shank @ [l_shank_quat[0], l_shank_quat[1], l_shank_quat[2]]
        T_l_shank = quaternion_to_matrix([-l_knee_quat[0], -l_knee_quat[2], -l_knee_quat[1], l_shank_quat[3]])
        T_l_shank_thigh = T_l_thigh.T @ T_l_shank
        l_knee = matrix_to_quaternion(T_l_shank_thigh)
        self.data.qpos[25] = abs(l_knee[3])
        # self.data.qpos[26] = l_knee[0]
        # self.data.qpos[27] = l_knee[1]
        self.data.qpos[28] = -abs(l_knee[2])  # 弯曲


        mujoco.mj_forward(self.model, self.data)


    if hasattr(self, 'viewer') and self.viewer:
        self.update_visualization()


def updateData(device):
    print(
        "ID:{}  {}{}  AccX:{}  AccY:{}  AccZ:{}  AsX:{}  AsY:{}  AsZ:{}  AngX:{}  AngY:{}  AngZ:{} "
        .format(device.get("CanID"), device.get("canmode_2"), device.get("canmode_1"),
                device.get("AccX"),device.get("AccY"),device.get("AccZ"),
                device.get("AsX"), device.get("AsY"), device.get("AsZ"),
                device.get("AngX"),device.get("AngY"), device.get("AngZ")))


if __name__ == "__main__":
    # 拿到设备模型 Get the device model
    # DeviceModel("设备名称可以自定义", "串口号/COM口号", "串口波特率", "CAN波特率(单位K)")
    device = DeviceModel("imu", "COM5", 2000000, 1000000, imu_data_solving)


    # 启动可视化
    device.start_visualization()


    # 开启设备 Turn on the device
    device.openDevice()

    # 读取回传速率
    # device.readReg(0x03)
    # time.sleep(1)

    # 设置加速度滤波和K值滤波
    #time.sleep(1)
    #device.writeReg(0x2A, 0x01F4)   # 默认为500 0x01F4
    #device.writeReg(0x25, 0x1E)   # 默认为30 0x1E

    # 设置20hz回传速率
    # device.writeReg(0x03, 7)

    # 设置透传模式
    # device.setET()

try:
    device.openDevice()
    # 主程序循环
    while True:
        time.sleep(1)  # 避免占用过多CPU
except KeyboardInterrupt:
    print("程序被用户中断")
    device.closeDevice()  # 确保设备正确关闭
