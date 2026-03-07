"""
imu数据转化为isaaclab中观测
"""

import torch
import numpy as np
import time
from datetime import datetime
import csv
from scipy.spatial.transform import Rotation

pelvis_pos_indices = 3
r_hip_pos_indices = 7
r_knee_pos_indices = 11
l_hip_pos_indices = 21
l_knee_pos_indices = 25

pelvis_vel_indices = 3
r_hip_vel_indices = 6
r_knee_vel_indices = 9
l_hip_vel_indices = 17
l_knee_vel_indices = 20

def imu_to_obs(pos, vel):
    joint_pos = np.concatenate([
        q_3dof(pos, r_hip_pos_indices),
        q_3dof(pos, l_hip_pos_indices),
        q_1dof(pos, r_knee_pos_indices),
        q_1dof(pos, l_knee_pos_indices),
    ])

    joint_vel = np.concatenate([
        [0, v_3dof(vel, r_hip_vel_indices)[1], 0],
        [0, v_3dof(vel, l_hip_vel_indices)[1], 0],
        v_1dof(vel, r_knee_vel_indices),
        v_1dof(vel, l_knee_vel_indices),
    ])

    # (w,x,y,z)
    root_pos_w = quaternion_to_tangent_and_normal(
        [pos[pelvis_pos_indices], pos[pelvis_pos_indices+1], pos[pelvis_pos_indices+2], pos[pelvis_pos_indices+3]])
    root_vel_w = np.array(v_3dof(vel, pelvis_vel_indices))
    # root_pos_w = [1, 0, 0, 0, 0, 1]
    # root_vel_w = [0,0,0]
    # print(root_pos_w, root_vel_w)

    return [joint_pos, joint_vel, root_pos_w, root_vel_w]


# 坐标变换
def q_3dof(data, indices):
    # mujoco(wxyz)-->world(xyz)球关节
    mujoco_data = quaternion_to_euler([data[indices+1], data[indices+2], data[indices+3], data[indices]])
    pos_data = [mujoco_data[0], -mujoco_data[2], mujoco_data[1]]  # (x,-z,y)
    return pos_data

def q_1dof(data, indices):
    # mujoco(wxyz)-->world(y)单关节
    mujoco_data = quaternion_to_euler([data[indices + 1], data[indices + 2], data[indices + 3], data[indices]])
    pos_data = [-mujoco_data[2]]  # (x,-z,y)
    return pos_data

def v_3dof(data, indices):
    vel_data = [data[indices], -data[indices+2], data[indices+1]]
    return vel_data

def v_1dof(data, indices):
    vel_data = [-data[indices+2]]
    return vel_data


def quaternion_rotate_vector(q: list[float], v: list[float]) -> list[float]:
    """
    用四元数旋转向量（核心：实现原quat_apply的功能）。
    参数：
        q: 四元数，格式 [w, x, y, z]（单位四元数）
        v: 3维向量，格式 [x, y, z]
    返回：
        旋转后的3维向量 [x', y', z']
    """
    w, x, y, z = q
    vx, vy, vz = v

    # 四元数与向量的乘法公式：v' = q * v * q⁻¹（q⁻¹为q的共轭，即[w, -x, -y, -z]）
    # 展开计算旋转后的向量分量
    x_out = (w ** 2 + x ** 2 - y ** 2 - z ** 2) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz
    y_out = 2 * (x * y + w * z) * vx + (w ** 2 - x ** 2 + y ** 2 - z ** 2) * vy + 2 * (y * z - w * x) * vz
    z_out = 2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (w ** 2 - x ** 2 - y ** 2 + z ** 2) * vz

    return [x_out, y_out, z_out]


def quaternion_to_tangent_and_normal(q: list[float]) -> list[float]:
    """
    输入4维四元数，输出6维向量（切向量+法向量）。
    参数：
        q: 四元数，格式 [w, x, y, z]（单位四元数）
    返回：
        6维向量 [tx, ty, tz, nx, ny, nz]，其中：
            tx, ty, tz：旋转后的切向量（原参考方向 (1,0,0)）
            nx, ny, nz：旋转后的法向量（原参考方向 (0,0,1)）
    """
    # 参考切向量（初始方向：x轴正方向 (1,0,0)）
    ref_tangent = [1.0, 0.0, 0.0]
    # 参考法向量（初始方向：z轴正方向 (0,0,1)）
    ref_normal = [0.0, 0.0, 1.0]

    # 用四元数旋转两个参考向量
    tangent = quaternion_rotate_vector(q, ref_tangent)  # 旋转后的切向量（3维）
    normal = quaternion_rotate_vector(q, ref_normal)  # 旋转后的法向量（3维）
    # print(tangent + normal)
    # 拼接为6维向量
    return tangent + normal  # 前3维切向量，后3维法向量


def quaternion_to_euler(quaternion):
    r = Rotation.from_quat(quaternion)
    euler = r.as_euler('XYZ', degrees=True) / 180 * np.pi       # 内旋 XYZ
    return euler