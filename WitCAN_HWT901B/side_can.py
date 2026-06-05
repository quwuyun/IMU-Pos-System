import os
import threading
import time
import can
import numpy as np
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

class CanConfig:
    # CAN 通道名称（在 Linux 上通常为 can0）
    channel = 'can1'
    # CAN 波特率（bit/s）
    bitrate = 1000000


class DeviceModel:

    # 设备名称
    deviceName = "imu"

    # 设备数据字典
    deviceData = {}

    # CAN 是否已打开
    isOpen = False

    # CAN 总线对象
    bus = None

    # CAN 配置
    canConfig = CanConfig()

    # 当前 CAN ID 模式信息
    canid = None

    # 当前正在读取的寄存器地址
    statReg = None

    def __init__(self, deviceName, channel, bitrate, callback_method):
        """
        :param deviceName: 设备名称（自定义）
        :param channel: CAN 通道名，例如 'can0'
        :param bitrate: CAN 波特率（bit/s），例如 1000000
        :param callback_method: 数据更新回调，形如 callback(device)
        """
        print("初始化基于 CAN 的设备模型")
        self.deviceName = deviceName
        self.canConfig.channel = channel
        self.canConfig.bitrate = bitrate
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


    # region 获取 / 设置设备数据
    def set(self, key, value):
        self.deviceData[key] = value

    def get(self, key):
        return self.deviceData.get(key, None)

    def remove(self, key):
        if key in self.deviceData:
            del self.deviceData[key]
    # endregion

    # 打开 CAN 设备
    def openDevice(self):
        """
        打开 CAN 总线，并启动后台读取线程
        """
        self.closeDevice()  # 确保之前的连接关闭

        try:
            self.bus = can.interface.Bus(
                interface='socketcan',
                channel=self.canConfig.channel,
                bitrate=self.canConfig.bitrate
            )
            self.isOpen = True
            print(f"CAN 总线 {self.canConfig.channel} 已打开，波特率 {self.canConfig.bitrate}")

            # 启动读取线程
            t = threading.Thread(target=self.readDataTh, args=("CAN-Data-Thread",), daemon=True)
            t.start()
            print("设备打开成功，开始监听 CAN 数据")
        except Exception as ex:
            print(f"打开 CAN 通道 {self.canConfig.channel} 失败：{ex}")
            self.isOpen = False

    # 关闭 CAN 设备
    def closeDevice(self):
        if self.bus is not None:
            try:
                self.bus.shutdown()
                print("CAN 总线已关闭")
            except Exception as ex:
                print(f"关闭 CAN 总线时出错：{ex}")
        self.isOpen = False
        self.bus = None
        print("设备关闭了")

    # CAN 监听线程
    def readDataTh(self, threadName):
        print("启动 " + threadName)
        while True:
            if not self.isOpen:
                print("CAN 未打开，停止监听线程")
                break

            try:
                # 超时时间 1 秒，与 can_reader.py 中保持一致
                msg = self.bus.recv(1.0)
                if msg is None:
                    print("未接收到 CAN 消息")
                    continue

                # 记录 CAN ID（仅用于显示）
                self.canid = msg.arbitration_id
                # 简单标记：假设为标准数据帧，后续可根据 msg.is_extended_id 等进行更精确判断
                self.canmode_1 = "数据帧"
                self.canmode_2 = "标准" if not msg.is_extended_id else "拓展"

                # 期望数据长度为 8 字节
                data = msg.data
                if len(data) != 8:
                    # 如果数据长度不是 8，可以根据你的协议决定是否忽略
                    print(f"收到非 8 字节数据帧，ID={hex(msg.arbitration_id)} len={len(data)}")
                    continue

                # 直接将 8 字节传给原来的解析函数
                self.processData(data)

            except KeyboardInterrupt:
                print("CAN 监听线程被中断")
                break
            except Exception as ex:
                print(f"CAN 监听线程异常：{ex}")
                time.sleep(0.1)

    # 数据解析
    def processData(self, Bytes):

        # 时间
        if Bytes[1] == 0x50:
            pass

        # 加速度
        elif Bytes[1] == 0x51:
            Ax = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 16
            Ay = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 16
            Az = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 16
            A_Gz = Az - g
            self.set("AccX", round(Ax, 3))
            self.set("AccY", round(Ay, 3))
            self.set("AccZ", round(Az, 3))
            self.A = [Ax, Ay, Az]
            # print("A:{}".format(self.A))
            self.A_G = [Ax, Ay, A_Gz]
            # print("A_G:{}".format(self.A_G))

            # 记录 CAN ID 和模式信息
            self.set("CanID", self.canid)
            self.set("canmode_1", self.canmode_1)
            self.set("canmode_2", self.canmode_2)

            # 回调通知上层
            if self.callback_method is not None:
                self.callback_method(self)

        # 角速度
        elif Bytes[1] == 0x52:
            Gx = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 2000
            Gy = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 2000
            Gz = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 2000
            self.set("AsX", round(Gx, 3))
            self.set("AsY", round(Gy, 3))
            self.set("AsZ", round(Gz, 3))
            self.G = [Gx, Gy, Gz]
            # print("G:{}".format(self.G))

        # 角度
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

        # 磁场
        elif Bytes[1] == 0x54:
            Hx = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) * 8.333 / 1000
            Hy = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) * 8.333 / 1000
            Hz = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) * 8.333 / 1000
            self.set("HX", round(Hx, 3))
            self.set("HY", round(Hy, 3))
            self.set("HZ", round(Hz, 3))

        # 读取回传
        elif Bytes[1] == 0x5F:
            value = self.getSignInt16(Bytes[3] << 8 | Bytes[2])
            self.set(str(self.statReg), value)
            print("读取数据返回  reg:{}   value:{} ".format(self.statReg, value))

        else:
            pass

    # 获得 int16 有符号数
    def getSignInt16(self, num):
        if num >= (1 << 15):
            num -= (1 << 16)
        return num


def imu_data_saving(self):
    # imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
    imu_data_list = [self.imu_data_dict[i] for i in range(0, 5)]
    self.R_shank, self.R_thigh, self.Pelvis, self.L_thigh, self.L_shank = imu_data_list

    print("imu_data_list:{}".format(imu_data_list))
    # print("R_shank:{}".format(self.R_shank))


if __name__ == "__main__":
    print("IMU 设备读取程序启动")

    # 初始化 CAN 接口（与 can_reader.py 保持一致）
    os.system('sudo ip link set can1 up type can bitrate 1000000')
    os.system('sudo ifconfig can1 txqueuelen 65536')

    # 创建设备实例
    device = DeviceModel("测试设备_CAN", "can0", 1000000, imu_data_saving)

    # 打开设备（开始后台监听 CAN 数据）
    device.openDevice()

    try:
        # 主循环保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序被用户中断")
        device.closeDevice()