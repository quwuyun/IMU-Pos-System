"""
使用异步循环读取多个IMU并解算关节数据
"""

import asyncio
from bleak import BleakScanner
import wt_device_model_hy
from scipy.spatial.transform import Rotation
import numpy as np
import mujoco  # pip install mujoco==2.3.7
import zmq  # pip install zmq

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

class WTMultiIMU:
    def __init__(self, mac_list):
        self.mac_list = mac_list
        self.imu_num = len(mac_list)

        # 数值结构：idx -> [quat_xyzw, gyro_xyz]
        self.imu_data_dict = {}
        for i in range(self.imu_num):
            self.imu_data_dict[i] = [[0, 0, 0, 1], [0, 0, 0]]

        self.calib_limit = 1000  # 标定的次数
        self.calib_count = 0
        self.calibrated = False

        # 世界坐标系到肢体坐标系转换
        self.pelvis_ori_init = []
        self.r_thigh_ori_init = []
        self.l_thigh_ori_init = []
        self.mount_matrix = np.array([
            [0, -1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])

        # mujoco
        self.model_file = './model/new/walk_new_quat_body0901.xml'
        self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
        self.data = mujoco.MjData(self.model)

        # zmq
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5555")

    # 回调函数
    def on_numeric(self, imu_idx, q_xyzw, g_xyz):
        self.imu_data_dict[imu_idx] = [q_xyzw, g_xyz]
        if not self.calibrated:
            self._calibration()
        else:
            self._compute_relative()
        # 打印
        # print("idx:", imu_idx, "data:", self.imu_data_dict[imu_idx])

    # 标定
    def _calibration(self):
        self.calib_count += 1

        if self.calib_count == self.calib_limit:
            imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
            # Pelvis, R_thigh, L_thigh, = imu_data_list
            Pelvis, R_thigh, = imu_data_list

            self.pelvis_ori_init = quaternion_to_matrix(Pelvis[0])
            self.r_thigh_ori_init = quaternion_to_matrix(R_thigh[0])
            # self.l_thigh_ori_init = quaternion_to_matrix(L_thigh[0])

            self.calibrated = True
            print("\n\nCalibration finished!!!\n\n")

    # 解算姿态
    def _compute_relative(self):
        # 三个IMU：骨盆，右大腿，左大腿
        imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        # Pelvis, R_thigh, L_thigh, = imu_data_list
        Pelvis, R_thigh, = imu_data_list

        """骨盆"""
        pelvis_quaternion = Pelvis[0]
        pelvis_ori = quaternion_to_matrix(pelvis_quaternion)
        Pelvis_ori = self.pelvis_ori_init.T @ pelvis_ori
        pelvis_quat = matrix_to_quaternion(Pelvis_ori)
        PELVIS_quat = self.mount_matrix @ [pelvis_quat[0], pelvis_quat[1], pelvis_quat[2]]
        self.data.qpos[3] = pelvis_quat[3]
        self.data.qpos[4] = -PELVIS_quat[0]  # 外摆
        self.data.qpos[5] = -PELVIS_quat[2]
        self.data.qpos[6] = -PELVIS_quat[1]  # 弯曲
        T_pelvis = quaternion_to_matrix([-PELVIS_quat[0], -PELVIS_quat[2], -PELVIS_quat[1], pelvis_quat[3]])
        # print(f"骨盆角度：{PELVIS_quat}")
        # 角速度
        pelvis_angle_velocity = self.mount_matrix @ Pelvis[1]
        qvel_pelvis = pelvis_angle_velocity
        self.data.qvel[3] = -qvel_pelvis[0]
        self.data.qvel[4] = -qvel_pelvis[2]
        self.data.qvel[5] = -qvel_pelvis[1]

        '''右髋'''
        r_thigh_quaternion = R_thigh[0]
        r_thigh_ori = quaternion_to_matrix(r_thigh_quaternion)
        R_thigh_ori = self.r_thigh_ori_init.T @ r_thigh_ori
        r_thigh_quat = matrix_to_quaternion(R_thigh_ori)
        r_hip_quat = self.mount_matrix @ [r_thigh_quat[0], r_thigh_quat[1], r_thigh_quat[2]]
        T_r_thigh = quaternion_to_matrix([-r_hip_quat[0], -r_hip_quat[2], -r_hip_quat[1], r_thigh_quat[3]])
        T_r_thigh_pelvis = T_pelvis.T @ T_r_thigh
        r_hip = matrix_to_quaternion(T_r_thigh_pelvis)
        self.data.qpos[7] = r_hip[3]
        self.data.qpos[8] = r_hip[0]  # 外摆
        self.data.qpos[9] = r_hip[1]
        self.data.qpos[10] = r_hip[2]  # 弯曲
        # 角速度
        r_thigh_angle_velocity = self.mount_matrix @ R_thigh[1]
        qvel_r_hip = r_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[6] = -qvel_r_hip[0]
        self.data.qvel[7] = -qvel_r_hip[2]
        self.data.qvel[8] = -qvel_r_hip[1]

        # '''左髋'''
        # l_thigh_quaternion = L_thigh[0]
        # l_thigh_ori = quaternion_to_matrix(l_thigh_quaternion)
        # L_thigh_ori = self.l_thigh_ori_init.T @ l_thigh_ori
        # l_thigh_quat = matrix_to_quaternion(L_thigh_ori)
        # l_hip_quat = self.mount_matrix @ [l_thigh_quat[0], l_thigh_quat[1], l_thigh_quat[2]]
        # T_l_thigh = quaternion_to_matrix([-l_hip_quat[0], -l_hip_quat[2], -l_hip_quat[1], l_thigh_quat[3]])
        # T_l_thigh_pelvis = T_pelvis.T @ T_l_thigh
        # l_hip = matrix_to_quaternion(T_l_thigh_pelvis)
        # self.data.qpos[21] = l_hip[3]
        # self.data.qpos[22] = l_hip[0]  # 外摆
        # self.data.qpos[23] = l_hip[1]
        # self.data.qpos[24] = l_hip[2]  # 弯曲
        # # 角速度
        # l_thigh_angle_velocity = self.mount_matrix @ L_thigh[1]
        # qvel_l_hip = l_thigh_angle_velocity - pelvis_angle_velocity
        # self.data.qvel[17] = -qvel_l_hip[0]
        # self.data.qvel[18] = -qvel_l_hip[2]
        # self.data.qvel[19] = -qvel_l_hip[1]

        print("右髋角度(deg):", qvel_pelvis, "右髋角速度:", qvel_r_hip)

        self.publisher.send_multipart([self.data.qpos.tobytes()])

    # 蓝牙扫描，连接
    async def run(self):
        # 扫描一次,避免 InProgress
        devices = await BleakScanner.discover(timeout=5.0)
        by_addr = {d.address.upper(): d for d in devices}

        missing = [m for m in self.mac_list if m not in by_addr]
        if missing:
            raise RuntimeError(f"Not found in scan: {missing}")

        # 并发连接
        tasks = []
        for i, mac in enumerate(self.mac_list):
            dev = by_addr[mac]
            dm = wt_device_model_hy.DeviceModel(
                deviceName=f"WT901_{i}",
                BLEDevice=dev,
                callback_method=None,
                imu_idx=i,
                callback_numeric=self.on_numeric,
            )
            tasks.append(asyncio.create_task(dm.openDevice()))

        await asyncio.gather(*tasks)


async def main():
    # IMU MAC地址
    mac_list = [
        "E6:4F:C8:40:5E:BB",
        "D7:5E:B3:78:1E:62",
    ]

    mgr = WTMultiIMU(mac_list)
    await mgr.run()

if __name__ == "__main__":
    asyncio.run(main())
