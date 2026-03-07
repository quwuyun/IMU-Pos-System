"""
用于windows系统上读取并解码多个LPMS-B2蓝牙IMU
IMU信息包含四元数和角速度，可在上位机设置选择
"""

import serial
import time
import struct
import numpy as np


class read_imu_multi:
    def __init__(self, ports, baudrate=921600):
        """
        windows系统上，初始化多串口LPMS-B2蓝牙IMU
        :param ports: 串口列表，例如 ['COM12', 'COM14']
        :param baudrate: 波特率，默认921600
        """
        self.imu_num = len(ports)  # IMU数量
        self.ports = ports
        self.baudrate = baudrate
        self.count = 0
        self.time = time.time()

        # 初始化串口对象
        self.ser_list = []
        self.buf_list = []  # 每个IMU的缓存
        for port in ports:
            try:
                ser = serial.Serial(port, baudrate, timeout=1 / 400)
                self.ser_list.append(ser)
                self.buf_list.append(b"")  # 每个串口对应一个缓存
                print(f"✅ 成功连接 {port}")
            except Exception as e:
                raise RuntimeError(f"❌ 连接{port}失败: {e}")

        # 初始化数据字典
        self.imu_data_dict = {}
        for i in range(self.imu_num):
            # 初始值[四元数, 角速度]
            self.imu_data_dict[i] = [[1, 0, 0, 0], [0, 0, 0]]

    def extract_frames(self, buf):
        """从单IMU缓存中提取完整帧"""
        frames = []
        while True:
            start = buf.find(b'\x3A')  # 包头0x3A
            if start < 0:
                break
            if len(buf) < start + 7:
                break

            # 解析数据长度
            data_len = int.from_bytes(buf[start + 5:start + 7], 'little')
            frame_len = 7 + data_len + 4  # header(7) + payload + LRC(2) + tail(2:0D0A)

            if len(buf) < start + frame_len:
                break

            frame = buf[start:start + frame_len]
            # 验证帧尾
            if frame.endswith(b'\x0D\x0A'):
                frames.append(frame)
                buf = buf[start + frame_len:]
            else:
                buf = buf[start + 1:]
        return frames, buf

    def parse_lpms_payload(self, payload_bytes):
        """解析单帧payload，返回{timestamp, gyro, quat}"""
        data = {}
        if len(payload_bytes) < 2:
            return data

        # 时间戳（uint16_t，1/400秒）
        timestamp_raw = int.from_bytes(payload_bytes[0:2], 'little')
        data["timestamp"] = timestamp_raw / 400.0

        # 解析float32小端数据
        floats = []
        for i in range(4, len(payload_bytes), 4):
            if i + 4 <= len(payload_bytes):
                val = struct.unpack('<f', payload_bytes[i:i + 4])[0]
                floats.append(val)
        try:
            # 88字节
            data["gyro"] = tuple(floats[0:3])  # 角速度 rad/s
            data["quat"] = tuple(floats[3:7])  # 四元数(w,x,y,z)

        except IndexError as e:
            print(f"⚠️ 解析异常: {e} | payload长度: {len(payload_bytes)}")
        return data

    def read_imu_data(self):
        """读取所有IMU数据，返回格式对齐485版本的字典"""
        for imu_idx in range(self.imu_num):
            ser = self.ser_list[imu_idx]
            buf = self.buf_list[imu_idx]

            chunk = ser.read(43)  # 单次读取足够多字节
            if chunk:
                buf += chunk
                # 提取完整帧
                frames, buf = self.extract_frames(buf)
                self.buf_list[imu_idx] = buf  # 更新缓存

                # 解析最新数据帧
                if frames:
                    latest_frame = frames[-1]  # 最后一帧是缓存中最新数据帧
                    data_len = int.from_bytes(latest_frame[5:7], 'little')
                    payload = latest_frame[7:7 + data_len]
                    parsed_data = self.parse_lpms_payload(payload)

                    # 更新数据
                    if all(key in parsed_data for key in ["gyro", "quat"]):
                        # q = list(parsed_data["quat"])  # 四元数
                        q = parsed_data["quat"]
                        q = [q[1], q[2], q[3], q[0]]
                        g = list(parsed_data["gyro"])  # 角速度
                        self.imu_data_dict[imu_idx] = [q, g]
        # print(self.imu_data_dict)

        # 计数和帧率打印
        # self.count += 1
        # if self.count % 1000 == 0:
        #     elapsed = time.time() - self.time
        #     print(f"计数:{self.count} | 耗时:{elapsed:.3f}s | 帧率:{1000 / elapsed:.1f}Hz")
        #     self.time = time.time()

        return self.imu_data_dict

    def close(self):
        """关闭所有串口"""
        for ser in self.ser_list:
            if ser.is_open:
                ser.close()
        print("\n🔌 所有串口已关闭")


if __name__=="__main__":
    # port1 = 'COM12'  # windows上
    # port2 = 'COM14'
    port1 = '/dev/rfcomm0'  # 树莓派上
    port2 = '/dev/rfcomm1'

    IMU_PORTS = [port1, port2]  # 蓝牙设置中的传出端口
    imu = read_imu_multi(ports=IMU_PORTS)
    while True:
        imu_data = imu.read_imu_data()
        for idx in range(imu.imu_num):
            q, g = imu_data[idx]
            print(f"IMU{idx}", f"角速度(rad/s): {g[0]:+.3f},{g[1]:+.3f},{g[2]:+.3f}",
                  f" 四元数: {q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f}")
        # time.sleep(0.001)

    # 单个imu测试
    # imu = read_imu()
    # imu.test()