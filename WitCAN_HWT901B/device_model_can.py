# coding:UTF-8
"""
基于 CAN 的 IMU 设备读取程序
不通过串口 / AT 指令，直接从 CAN 总线读取原始数据帧并解析。
"""

import threading
import time
import can


class CanConfig:
    """
    CAN 配置
    """
    # CAN 通道名称（在 Linux 上通常为 can0）
    channel = 'can1'
    # CAN 波特率（bit/s）
    bitrate = 1000000


class DeviceModel:
    """
    设备模型：通过 CAN 直接获取 IMU 数据
    """

    # 设备名称
    deviceName = "Pelvis_imu"

    # 设备数据字典
    deviceData = {}

    # CAN 是否已打开
    isOpen = False

    # CAN 总线对象
    bus = None

    # CAN 配置
    canConfig = CanConfig()

    # 当前 CAN ID 模式信息（用于显示）
    canid = None
    canmode_1 = ""  # 数据帧 / 远程帧（这里暂不区分，可根据需要扩展）
    canmode_2 = ""  # 标准 / 拓展（这里暂不区分，可根据需要扩展）

    # 当前正在读取的寄存器地址（如果后续扩展寄存器读写时用）
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
                    # 未接收到数据，可以按需注释掉这句，以免打印过多
                    # print("未接收到 CAN 消息")
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

    # 数据解析（沿用原 device_model.py 中的逻辑）
    def processData(self, Bytes):
        """
        Bytes: 长度为 8 的 bytearray / bytes
        协议约定：
            Bytes[1] 为数据类型：
                0x50: 时间（当前未处理）
                0x51: 加速度
                0x52: 角速度
                0x53: 角度
                0x54: 磁场
                0x5F: 寄存器读取回传
        """
        # 时间
        if Bytes[1] == 0x50:
            pass

        # 加速度
        elif Bytes[1] == 0x51:
            Ax = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 16
            Ay = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 16
            Az = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 16
            self.set("AccX", round(Ax, 3))
            self.set("AccY", round(Ay, 3))
            self.set("AccZ", round(Az, 3))

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

        # 角度
        elif Bytes[1] == 0x53:
            AngX = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 180
            AngY = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 180
            AngZ = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 180
            self.set("AngX", round(AngX, 3))
            self.set("AngY", round(AngY, 3))
            self.set("AngZ", round(AngZ, 3))

        # 磁场
        elif Bytes[1] == 0x54:
            Hx = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) * 8.333 / 1000
            Hy = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) * 8.333 / 1000
            Hz = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) * 8.333 / 1000
            self.set("HX", round(Hx, 3))
            self.set("HY", round(Hy, 3))
            self.set("HZ", round(Hz, 3))

        # 读取回传（如果你有通过 CAN 的寄存器读写协议，可以在这里扩展）
        elif Bytes[1] == 0x5F:
            value = self.getSignInt16(Bytes[3] << 8 | Bytes[2])
            self.set(str(self.statReg), value)
            print("读取数据返回  reg:{}   value:{} ".format(self.statReg, value))

        else:
            # 未知类型，可按需打印或忽略
            pass

    # 获得 int16 有符号数
    def getSignInt16(self, num):
        if num >= (1 << 15):
            num -= (1 << 16)
        return num


# 一个简单的回调函数，模仿原来的 updateData
def updateData(device: DeviceModel):
    print(
        "ID:{}  {}{}  AccX:{}  AccY:{}  AccZ:{}  AsX:{}  AsY:{}  AsZ:{}  AngX:{}  AngY:{}  AngZ:{}  Hx:{}  Hy:{}  Hz:{}"
        .format(
            device.get("CanID"),
            device.get("canmode_2") or "",
            device.get("canmode_1") or "",
            device.get("AccX"),
            device.get("AccY"),
            device.get("AccZ"),
            device.get("AsX"),
            device.get("AsY"),
            device.get("AsZ"),
            device.get("AngX"),
            device.get("AngY"),
            device.get("AngZ"),
            device.get("HX"),
            device.get("HY"),
            device.get("HZ"),
        )
    )


if __name__ == "__main__":
    import os

    print("基于 CAN 的 IMU 设备读取程序启动")

    # 初始化 CAN 接口（与 can_reader.py 保持一致）
    os.system('sudo ip link set can1 up type can bitrate 1000000')
    os.system('sudo ifconfig can1 up')

    # 创建设备实例
    device = DeviceModel("测试设备_CAN", "can1", 1000000, updateData)

    # 打开设备（开始后台监听 CAN 数据）
    device.openDevice()

    try:
        # 主循环保持运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序被用户中断")
        device.closeDevice()