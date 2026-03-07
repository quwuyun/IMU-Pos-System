"""
LPMS-B2蓝牙IMU解算人体下肢关节姿态(windows系统)
"""

import serial
import time
import struct
import zmq
import csv
import numpy as np
import mujoco       # pip install mujoco==2.3.7    pip install mujoco-python-viewer
from mujoco_viewer.mujoco_viewer import *
from scipy.spatial.transform import Rotation
# import linuxfd,signal,select

R_thigh = [[0, 0, 0, 1], [0, 0, 0]]
R_shank = [[0, 0, 0, 1], [0, 0, 0]]
R_foot = [[0, 0, 0, 1], [0, 0, 0]]
L_thigh = [[0, 0, 0, 1], [0, 0, 0]]
L_shank = [[0, 0, 0, 1], [0, 0, 0]]
L_foot = [[0, 0, 0, 1], [0, 0, 0]]
Pelvis = [[0, 0, 0, 1], [0, 0, 0]]

ENABLE_VIEWER = True

def quaternion_to_matrix(quaternion):
    # 创建四元数对象(x, y, z, w)
    r = Rotation.from_quat(quaternion)
    # 将四元数转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def matrix_to_quaternion(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat(canonical=False)
    return quaternion


class imu_posture:
    def __init__(self, ports, baudrate=115200):
        self.imu_num = len(ports)  # IMU数量
        self.ports = ports
        self.baudrate = baudrate
        self.ser_list = []
        self.buf_list = []
        for port in ports:
            try:
                ser = serial.Serial(port, baudrate, timeout=0.001)
                ser.reset_input_buffer()  # 清空缓存，避免旧数据堆积
                self.ser_list.append(ser)
                self.buf_list.append(b"")  # 每个串口对应一个缓存
                print(f"成功连接 {port}")
            except Exception as e:
                raise RuntimeError(f"连接{port}失败: {e}")

        self.imu_data_dict = {}
        for i in range(self.imu_num):
            self.imu_data_dict[i] = [[0, 0, 0, 1], [0, 0, 0]]
        # 解算相关中间变量
        self.r_thigh_ori_init = []  # 初始化的参考旋转矩阵
        self.r_shank_ori_init = []
        self.r_foot_ori_init = []
        self.l_thigh_ori_init = []
        self.l_shank_ori_init = []
        self.l_foot_ori_init = []
        self.pelvis_ori_init = []
        self.rot_r_hip = np.loadtxt(r'calibration/rot_r_hip.txt', delimiter=',')
        self.rot_l_hip = np.loadtxt(r'calibration/rot_l_hip.txt', delimiter=',')
        self.rot_r_shank = np.loadtxt(r'calibration/rot_r_shank.txt', delimiter=',')
        self.rot_l_shank = np.loadtxt(r'calibration/rot_l_shank.txt', delimiter=',')
        self.rot_r_ankle = np.loadtxt(r'calibration/rot_r_ankle.txt', delimiter=',')
        self.rot_l_ankle = np.loadtxt(r'calibration/rot_l_ankle.txt', delimiter=',')
        self.rot_pelvis = np.loadtxt(r'calibration/rot_pelvis.txt', delimiter=',')
        self.rot_test = np.loadtxt(r'calibration/rot_test.txt', delimiter=',')

        # self.rot_pelvis = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        # self.rot_r_hip = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        # self.rot_r_shank = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        # self.rot_l_hip = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        # self.rot_l_shank = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        # self.rot_r_ankle = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])
        # self.rot_l_ankle = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])

        self.count = 0
        self.time = time.time()
        self.time_last = time.time()
        self.last_time = time.time()
        self.det_t = 0.01
        self.last_qvel = np.zeros(15)
        self.LIST = np.zeros(50)
        self.output = []

        self.model_file = './model/new/walk_new_quat_body0901.xml'
        self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
        self.data = mujoco.MjData(self.model)
        self.viewer = MujocoViewer(self.model, self.data, width=1280, height=1024)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30
        self.viewer.cam.distance = 3.0
        self.viewer.cam.lookat = [0, 0, 1]

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
            print(f"解析异常: {e} | payload长度: {len(payload_bytes)}")
        return data

    def read_imu_data(self):
        """读取所有IMU数据，返回格式对齐485版本的字典"""
        for imu_idx in range(self.imu_num):
            ser = self.ser_list[imu_idx]
            buf = self.buf_list[imu_idx]

            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
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
                        q = [q[1], q[2], q[3], q[0]]  # (x,y,z,w)
                        g = list(parsed_data["gyro"])  # 角速度
                        self.imu_data_dict[imu_idx] = [q, g]
        # print(self.imu_data_dict)

        # 计数和帧率打印
        self.count += 1
        # if self.count % 1000 == 0:
        #     elapsed = time.time() - self.time
        #     print(f"计数:{self.count} | 耗时:{elapsed:.3f}s | 帧率:{1000 / elapsed:.1f}Hz")
        #     self.time = time.time()
        # time.sleep(0.001)

        return self.imu_data_dict

    def close(self):
        """关闭所有串口"""
        for ser in self.ser_list:
            if ser.is_open:
                ser.close()
        print("\n🔌 所有串口已关闭")

    # 初始标定
    def imu_calibration(self):
        imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        # Pelvis, R_thigh, L_thigh, R_shank, L_shank, R_foot, L_foot, = imu_data_list
        Pelvis, R_thigh, L_thigh, R_shank, L_shank, = imu_data_list
        # Pelvis, R_thigh, = imu_data_list
        # R_thigh, R_shank, = imu_data_list
        # R_shank, R_foot, = imu_data_list

        r_thigh_quaternion_init = R_thigh[0]
        r_shank_quaternion_init = R_shank[0]
        r_foot_quaternion_init = R_foot[0]
        l_thigh_quaternion_init = L_thigh[0]
        l_shank_quaternion_init = L_shank[0]
        l_foot_quaternion_init = L_foot[0]
        pelvis_quaternion_init = Pelvis[0]

        self.r_thigh_ori_init = quaternion_to_matrix(r_thigh_quaternion_init)
        self.r_shank_ori_init = quaternion_to_matrix(r_shank_quaternion_init)
        self.r_foot_ori_init = quaternion_to_matrix(r_foot_quaternion_init)
        self.l_thigh_ori_init = quaternion_to_matrix(l_thigh_quaternion_init)
        self.l_shank_ori_init = quaternion_to_matrix(l_shank_quaternion_init)
        self.l_foot_ori_init = quaternion_to_matrix(l_foot_quaternion_init)
        self.pelvis_ori_init = quaternion_to_matrix(pelvis_quaternion_init)

    # 记录原始数据
    # def get_rawdata(self):
    #     imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
    #     # Pelvis, R_thigh, L_thigh, R_shank, L_shank, R_foot, L_foot, = imu_data_list
    #     Pelvis, R_thigh, = imu_data_list
    #
    #     self.det_t = time.time() - self.last_time
    #     self.last_time = time.time()
    #     # print(self.det_t)
    #
    #     self.last_qvel[0] = Pelvis[1][0]
    #     self.last_qvel[1] = Pelvis[1][1]
    #     self.last_qvel[2] = Pelvis[1][2]
    #     self.LIST[0] = Pelvis[3][0]
    #     self.LIST[1] = Pelvis[3][1]
    #     self.LIST[2] = Pelvis[3][2]
    #     self.LIST[3] = Pelvis[1][0]
    #     self.LIST[4] = Pelvis[1][1]
    #     self.LIST[5] = Pelvis[1][2]
    #
    #     self.last_qvel[3] = R_thigh[1][0]
    #     self.last_qvel[4] = R_thigh[1][1]
    #     self.last_qvel[5] = R_thigh[1][2]
    #     self.LIST[10] = R_thigh[3][0]
    #     self.LIST[11] = R_thigh[3][1]
    #     self.LIST[12] = R_thigh[3][2]
    #     self.LIST[13] = R_thigh[1][0]
    #     self.LIST[14] = R_thigh[1][1]
    #     self.LIST[15] = R_thigh[1][2]
    #
    #     self.last_qvel[6] = R_shank[1][0]
    #     self.last_qvel[7] = R_shank[1][1]
    #     self.last_qvel[8] = R_shank[1][2]
    #     self.LIST[20] = R_shank[3][0]
    #     self.LIST[21] = R_shank[3][1]
    #     self.LIST[22] = R_shank[3][2]
    #     self.LIST[23] = R_shank[1][0]
    #     self.LIST[24] = R_shank[1][1]
    #     self.LIST[25] = R_shank[1][2]
    #
    #     self.last_qvel[9] = L_thigh[1][0]
    #     self.last_qvel[10] = L_thigh[1][1]
    #     self.last_qvel[11] = L_thigh[1][2]
    #     self.LIST[30] = L_thigh[3][0]
    #     self.LIST[31] = L_thigh[3][1]
    #     self.LIST[32] = L_thigh[3][2]
    #     self.LIST[33] = L_thigh[1][0]
    #     self.LIST[34] = L_thigh[1][1]
    #     self.LIST[35] = L_thigh[1][2]
    #
    #     self.last_qvel[12] = L_shank[1][0]
    #     self.last_qvel[13] = L_shank[1][1]
    #     self.last_qvel[14] = L_shank[1][2]
    #     self.LIST[40] = L_shank[3][0]
    #     self.LIST[41] = L_shank[3][1]
    #     self.LIST[42] = L_shank[3][2]
    #     self.LIST[43] = L_shank[1][0]
    #     self.LIST[44] = L_shank[1][1]
    #     self.LIST[45] = L_shank[1][2]
    #
    #     self.LIST = np.array(self.LIST)
    #     self.output.append(self.LIST)

    # imu数据解算
    def imu_data_solving(self):
        imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        # Pelvis, R_thigh, L_thigh, R_shank, L_shank, R_foot, L_foot, = imu_data_list
        Pelvis, R_thigh, L_thigh, R_shank, L_shank, = imu_data_list
        # Pelvis, R_thigh, = imu_data_list
        # R_thigh, R_shank, = imu_data_list
        # R_shank, R_foot, = imu_data_list

        # X向前，Y向左，Z向上
        # Position = camera[0]
        # data.qpos[0] = -Position.z
        # data.qpos[1] = -Position.x
        # data.qpos[2] = Position.y
        pelvis_quaternion = Pelvis[0]
        pelvis_ori = quaternion_to_matrix(pelvis_quaternion)
        Pelvis_ori = self.pelvis_ori_init.T @ pelvis_ori
        pelvis_quat = matrix_to_quaternion(Pelvis_ori)
        PELVIS_quat = self.rot_pelvis @ [pelvis_quat[0], pelvis_quat[1], pelvis_quat[2]]
        self.data.qpos[3] = pelvis_quat[3]
        self.data.qpos[4] = -PELVIS_quat[0]  # 外摆
        self.data.qpos[5] = -PELVIS_quat[2]
        self.data.qpos[6] = -PELVIS_quat[1]  # 弯曲
        T_pelvis = quaternion_to_matrix([-PELVIS_quat[0], -PELVIS_quat[2], -PELVIS_quat[1], pelvis_quat[3]])
        # print(f"骨盆角度：{PELVIS_quat}")
        '''角速度'''
        pelvis_angle_velocity = self.rot_pelvis @ Pelvis[1]
        qvel_pelvis = pelvis_angle_velocity
        self.data.qvel[3] = -qvel_pelvis[0]
        self.data.qvel[4] = -qvel_pelvis[2]
        self.data.qvel[5] = -qvel_pelvis[1]

        '''右髋关节角度'''
        r_thigh_quaternion = R_thigh[0]
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
        '''角速度'''
        r_thigh_angle_velocity = self.rot_r_hip @ R_thigh[1]
        qvel_r_hip = r_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[6] = -qvel_r_hip[0]
        self.data.qvel[7] = -qvel_r_hip[2]
        self.data.qvel[8] = -qvel_r_hip[1]

        '''右膝关节角度'''
        r_shank_quaternion = R_shank[0]
        r_shank_ori = quaternion_to_matrix(r_shank_quaternion)
        R_shank_ori = self.r_shank_ori_init.T @ r_shank_ori
        r_shank_quat = matrix_to_quaternion(R_shank_ori)
        r_knee_quat = self.rot_r_shank @ [r_shank_quat[0], r_shank_quat[1], r_shank_quat[2]]
        T_r_shank = quaternion_to_matrix([-r_knee_quat[0], -r_knee_quat[2], -r_knee_quat[1], r_shank_quat[3]])
        T_r_shank_thigh = T_r_thigh.T @ T_r_shank
        r_knee = matrix_to_quaternion(T_r_shank_thigh)
        self.data.qpos[11] = abs(r_knee[3])
        # data.qpos[12] = r_knee[0]
        # data.qpos[13] = r_knee[1]
        self.data.qpos[14] = -abs(r_knee[2])  # 弯曲
        '''角速度'''
        r_shank_angle_velocity = self.rot_r_shank @ R_shank[1]
        qvel_r_knee = r_shank_angle_velocity[1] - r_thigh_angle_velocity[1]
        self.data.qvel[11] = -qvel_r_knee

        '''右踝关节角度'''
        r_foot_quaternion = R_foot[0]
        r_foot_ori = quaternion_to_matrix(r_foot_quaternion)
        R_foot_ori = self.r_foot_ori_init.T @ r_foot_ori
        r_foot_quat = matrix_to_quaternion(R_foot_ori)
        r_ankle_quat = self.rot_r_ankle @ [r_foot_quat[0], r_foot_quat[1], r_foot_quat[2]]
        T_r_foot = quaternion_to_matrix([-r_ankle_quat[0], -r_ankle_quat[2], -r_ankle_quat[1], r_foot_quat[3]])
        T_r_foot_shank = T_r_shank.T @ T_r_foot
        r_ankle = matrix_to_quaternion(T_r_foot_shank)
        self.data.qpos[15] = r_ankle[3]
        self.data.qpos[16] = r_ankle[0]  # 外摆
        self.data.qpos[17] = r_ankle[1]
        self.data.qpos[18] = r_ankle[2]  # 弯曲
        '''角速度'''
        r_foot_angle_velocity = self.rot_r_ankle @ R_foot[1]
        qvel_r_ankle = r_foot_angle_velocity - r_shank_angle_velocity
        self.data.qvel[12] = -qvel_r_ankle[0]
        self.data.qvel[13] = -qvel_r_ankle[2]
        self.data.qvel[14] = -qvel_r_ankle[1]

        '''左髋关节角度'''
        l_thigh_quaternion = L_thigh[0]
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
        '''角速度'''
        l_thigh_angle_velocity = self.rot_l_hip @ L_thigh[1]
        qvel_l_hip = l_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[17] = -qvel_l_hip[0]
        self.data.qvel[18] = -qvel_l_hip[2]
        self.data.qvel[19] = -qvel_l_hip[1]

        '''左膝关节角度'''
        l_shank_quaternion = L_shank[0]
        l_shank_ori = quaternion_to_matrix(l_shank_quaternion)
        L_shank_ori = self.l_shank_ori_init.T @ l_shank_ori
        l_shank_quat = matrix_to_quaternion(L_shank_ori)
        l_knee_quat = self.rot_l_shank @ [l_shank_quat[0], l_shank_quat[1], l_shank_quat[2]]
        T_l_shank = quaternion_to_matrix([-l_knee_quat[0], -l_knee_quat[2], -l_knee_quat[1], l_shank_quat[3]])
        T_l_shank_thigh = T_l_thigh.T @ T_l_shank
        l_knee = matrix_to_quaternion(T_l_shank_thigh)
        self.data.qpos[25] = abs(l_knee[3])
        # data.qpos[26] = l_knee[0]
        # data.qpos[27] = l_knee[1]
        self.data.qpos[28] = -abs(l_knee[2])  # 弯曲
        '''角速度'''
        l_shank_angle_velocity = self.rot_l_shank @ L_shank[1]
        qvel_l_knee = l_shank_angle_velocity[1] - l_thigh_angle_velocity[1]
        self.data.qvel[22] = -qvel_l_knee

        '''左踝关节角度'''
        l_foot_quaternion = L_foot[0]
        l_foot_ori = quaternion_to_matrix(l_foot_quaternion)
        L_foot_ori = self.l_foot_ori_init.T @ l_foot_ori
        l_foot_quat = matrix_to_quaternion(L_foot_ori)
        l_ankle_quat = self.rot_l_ankle @ [l_foot_quat[0], l_foot_quat[1], l_foot_quat[2]]
        T_l_foot = quaternion_to_matrix([-l_ankle_quat[0], -l_ankle_quat[2], -l_ankle_quat[1], l_foot_quat[3]])
        T_l_foot_shank = T_l_shank.T @ T_l_foot
        l_ankle = matrix_to_quaternion(T_l_foot_shank)
        self.data.qpos[29] = l_ankle[3]
        self.data.qpos[30] = l_ankle[0]  # 外摆
        self.data.qpos[31] = l_ankle[1]
        self.data.qpos[32] = l_ankle[2]  # 弯曲
        '''角速度'''
        l_foot_angle_velocity = self.rot_l_ankle @ L_foot[1]
        qvel_l_ankle = l_foot_angle_velocity - l_shank_angle_velocity
        self.data.qvel[23] = -qvel_l_ankle[0]
        self.data.qvel[24] = -qvel_l_ankle[2]
        self.data.qvel[25] = -qvel_l_ankle[1]

        mujoco.mj_forward(self.model, self.data)

        if ENABLE_VIEWER:
            if self.viewer.is_alive:
                # viewer.cam.lookat = data.body('pelvis').subtree_com
                self.viewer.render()


if __name__ == '__main__':
    print('process init')
    IMU_PORTS = ['COM7','COM10','COM11','COM14','COM16']
    read_imu = imu_posture(ports=IMU_PORTS)

    # context = zmq.Context()
    # publisher = context.socket(zmq.PUB)
    # publisher.bind("tcp://*:5555")

    while True:
        read_imu.read_imu_data()
        if read_imu.count < 1000:
            read_imu.imu_calibration()  # 初始化
        elif read_imu.count == 1000:
            print('calibration')
        # elif 300 < read_imu.count < 1000:
        #     read_imu.get_rawdata()  # 记录数据
        # elif read_imu.count == 1000:
        #     read_imu.calibration_position()
        #     print('start')
        #     # threading.Thread(target=read_imu.calibration_position).start()
        else:
            # print('solving')
            read_imu.imu_data_solving()  # 正常工作
            # publisher.send_multipart([read_imu.data.qpos.tobytes(), read_imu.data.qvel.tobytes()])


    # # 定时器
    # # create special file objects
    # efd = linuxfd.eventfd(initval=0, nonBlocking=True)
    # sfd = linuxfd.signalfd(signalset={signal.SIGINT}, nonBlocking=True)
    # tfd = linuxfd.timerfd(rtc=True, nonBlocking=True)
    # # program timer and mask SIGINT
    # tfd.settime(1, 0.01)  # 第一次定时器间隔和以后所有定时器间隔
    # signal.pthread_sigmask(signal.SIG_SETMASK, {signal.SIGINT})
    # # create epoll instance and register special files
    # epl = select.epoll()
    # epl.register(efd.fileno(), select.EPOLLIN)
    # epl.register(sfd.fileno(), select.EPOLLIN)
    # epl.register(tfd.fileno(), select.EPOLLIN)
    # # start main loop
    # isrunning = True
    # while isrunning:
    #     # block until epoll detects changes in the registered files
    #     events = epl.poll(-1)
    #     t = time.time()
    #     # iterate over occurred events
    #     for fd, event in events:
    #         if fd == efd.fileno() and event & select.EPOLLIN:
    #             # event file descriptor readable: read and exit loop
    #             print("{0:.3f}: event file received update, exiting...".format(t))
    #             efd.read()
    #             isrunning = False
    #         elif fd == sfd.fileno() and event & select.EPOLLIN:
    #             # signal file descriptor readable: write to event file
    #             siginfo = sfd.read()
    #             if siginfo["signo"] == signal.SIGINT:
    #                 print("{0:.3f}: SIGINT received, notifying event file".format(t))
    #                 efd.write(1)
    #         elif fd == tfd.fileno() and event & select.EPOLLIN:
    #             # timer file descriptor readable: display that timer has expired
    #             tfd.read()
    #             read_imu.read_imu_data_right()
    #             read_imu.read_imu_data_left()
    #             if read_imu.count < 300:
    #                 read_imu.imu_calibration()  # 初始化
    #             elif read_imu.count == 300:
    #                 print('calibration')
    #             # elif 300 < read_imu.count < 1000:
    #             #     read_imu.get_rawdata()  # 记录数据
    #             # elif read_imu.count == 1000:
    #             #     read_imu.calibration_position()
    #             #     print('start')
    #             #     # threading.Thread(target=read_imu.calibration_position).start()
    #             else:
    #                 read_imu.imu_data_solving()  # 正常工作
    #                 publisher.send_multipart([read_imu.data.qpos.tobytes()])







