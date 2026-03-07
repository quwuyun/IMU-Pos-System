import numpy as np
import math
import time
from datetime import datetime
import csv
import threading
from scipy.spatial.transform import Rotation
from imu_485_usb_new import imu485


def imu_data_485_usb(name1, name2, name3, name4, name5, name6, port, imu_num, ):
    data_imu = imu485(port, imu_num)
    while True:
        data = data_imu.read_imu_data()
        for i in range(4):
            name1[i] = data[0][i]
            name2[i] = data[1][i]
            name3[i] = data[2][i]
            name4[i] = data[3][i]
            name5[i] = data[4][i]
            name6[i] = data[5][i]
        # print(data)


def quaternion_to_euler(quaternion):
    r = Rotation.from_quat(quaternion)
    euler = r.as_euler('XYZ', degrees=True) / 180 * np.pi       # 内旋 XYZ
    return euler


def euler_to_quaternion(euler):
    r = Rotation.from_euler('XYZ', euler / np.pi * 180, degrees=True)
    quaternion = r.as_quat()
    return quaternion


def EulerAnglesToRotationMat(theta):  # 动系 内旋 ZYX 右乘
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def RotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def EulerAnglesToQuat(intput_data):
    a = intput_data[0]
    b = intput_data[1]
    c = intput_data[2]

    sina = math.sin(a / 2)
    sinb = math.sin(b / 2)
    sinc = math.sin(c / 2)
    cosa = math.cos(a / 2)
    cosb = math.cos(b / 2)
    cosc = math.cos(c / 2)

    w = cosa * cosb * cosc + sina * sinb * sinc
    x = sina * cosb * cosc - cosa * sinb * sinc
    y = cosa * sinb * cosc + sina * cosb * sinc
    z = cosa * cosb * sinc - sina * sinb * cosc
    return [w, x, y, z]


def RotationMatrixToAxisAngle(R):
    global n_x, n_y, n_z, angle
    if (R[0][0] + R[1][1] + R[2][2] - 1) / 2 < 1:
        angle = math.acos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
        n_x = (R[2][1] - R[1][2]) / (2 * math.sin(angle))
        n_y = (R[0][2] - R[2][0]) / (2 * math.sin(angle))
        n_z = (R[1][0] - R[0][1]) / (2 * math.sin(angle))
    else:
        n_x, n_y, n_z, angle = 0, 0, 0, 0
    return n_x, n_y, n_z, angle


def quat_to_pos_matrix(x, y, z, w):
    # 创建位姿矩阵，写入位置
    # T = np.matrix([[0, 0, 0, p_x], [0, 0, 0, p_y], [0, 0, 0, p_z], [0, 0, 0, 1]])
    T = np.zeros((3, 3))
    T[0, 0] = 1 - 2 * pow(y, 2) - 2 * pow(z, 2)
    T[0, 1] = 2 * (x * y - w * z)
    T[0, 2] = 2 * (x * z + w * y)

    T[1, 0] = 2 * (x * y + w * z)
    T[1, 1] = 1 - 2 * pow(x, 2) - 2 * pow(z, 2)
    T[1, 2] = 2 * (y * z - w * x)

    T[2, 0] = 2 * (x * z - w * y)
    T[2, 1] = 2 * (y * z + w * x)
    T[2, 2] = 1 - 2 * pow(x, 2) - 2 * pow(y, 2)
    return T


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


def pos_matrix_to_quat(T):
    global x, y, z, w
    r11 = T[0, 0]
    r12 = T[0, 1]
    r13 = T[0, 2]
    r21 = T[1, 0]
    r22 = T[1, 1]
    r23 = T[1, 2]
    r31 = T[2, 0]
    r32 = T[2, 1]
    r33 = T[2, 2]
    helper = np.array(np.abs([r11 + r22 + r33, r11 - r22 - r33, -r11 + r22 - r33, -r11 - r22 + r33]))
    pos = np.argmax(helper)
    if pos == 0:
        w = (1 / 2) * math.sqrt(abs(1 + r11 + r22 + r33))
        x = (r32 - r23) / (4 * w)
        y = (r13 - r31) / (4 * w)
        z = (r21 - r12) / (4 * w)
    elif pos == 1:
        x = (1 / 2) * math.sqrt(abs(1 + r11 - r22 - r33))
        w = (r32 - r23) / (4 * x)
        y = (r21 + r12) / (4 * x)
        z = (r13 - r31) / (4 * x)
    elif pos == 2:
        y = (1 / 2) * math.sqrt(abs(1 - r11 + r22 - r33))
        w = (r13 - r31) / (4 * y)
        y = (r12 + r21) / (4 * y)
        z = (r23 + r32) / (4 * y)
    elif pos == 3:
        z = (1 / 2) * math.sqrt(abs(1 - r11 - r22 + r33))
        w = (r21 - r12) / (4 * z)
        y = (r13 + r31) / (4 * z)
        z = (r23 + r32) / (4 * z)
    return x, y, z, w


def imu_data(name, n):
    NAME = witsensor(n)
    # count = 0
    while True:
        name_imu = NAME.getdata()
        name[0] = name_imu
        # count += 1


def cam_data(name):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.pose)
    pipe.start(cfg)
    while True:
        frames = pipe.wait_for_frames()
        pose = frames.get_pose_frame()

        data = pose.get_pose_data()
        # 数据提取
        Position = data.translation
        Velocity = data.velocity
        Acceleration = data.acceleration
        rotation = data.rotation
        angular_velocity = data.angular_velocity
        angular_acceleration = data.angular_acceleration
        name[0] = [Position, Velocity, rotation, angular_velocity, Acceleration, angular_acceleration]
        # print(name[0][0])


def B_imu_data(name, n):
    NAME = bwitsensor(n)
    # count = 0
    while True:
        name_imu = NAME.getdata()
        name[0] = name_imu


def get_pelvis_ori(pose):
    data = pose.get_pose_data()
    w = data.rotation.w
    x = -data.rotation.z
    y = data.rotation.x
    z = -data.rotation.y
    # pitch = -math.asin(2.0 * (x * z - w * y))  # X轴
    # yaw = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)  # Y轴
    # roll = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)  # Z轴
    # pelvis_ori = EulerAnglesToRotationMat([pitch, yaw, roll])
    pelvis_ori = quat_to_pos_matrix(x, y, z, w)
    return pelvis_ori


# def calibration_pelvis_ori(ori_init):
#     number = 0
#     while True:
#         frames = pipe.wait_for_frames()
#         pose = frames.get_pose_frame()
#         while not pose:
#             frames = pipe.wait_for_frames()
#             pose = frames.get_pose_frame()
#         ori_init[0] = get_pelvis_ori(pose)
#         number += 1
#         if number > 500:
#             break
#     return ori_init


def calibration_pelvis_ori(ori_init):
    number = 0
    while True:
        Pelvis_quaternion = Pelvis[0]
        ori_init[0] = quaternion_to_matrix(Pelvis_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_r_thigh_ori(ori_init):
    number = 0
    while True:
        R_thigh_quaternion = R_thigh[0]
        ori_init[1] = quaternion_to_matrix(R_thigh_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_r_shank_ori(ori_init):
    number = 0
    while True:
        R_shank_quaternion = R_shank[0]
        ori_init[2] = quaternion_to_matrix(R_shank_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_r_arm_ori(ori_init):
    number = 0
    while True:
        R_arm_quaternion = R_arm[0]
        ori_init[3] = quaternion_to_matrix(R_arm_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_r_forearm_ori(ori_init):
    number = 0
    while True:
        R_forearm_quaternion = R_forearm[0]
        ori_init[4] = quaternion_to_matrix(R_forearm_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_l_thigh_ori(ori_init):
    number = 0
    while True:
        L_thigh_quaternion = L_thigh[0]
        ori_init[5] = quaternion_to_matrix(L_thigh_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_l_shank_ori(ori_init):
    number = 0
    while True:
        L_shank_quaternion = L_shank[0]
        ori_init[6] = quaternion_to_matrix(L_shank_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_l_arm_ori(ori_init):
    number = 0
    while True:
        L_arm_quaternion = L_arm[0]
        ori_init[7] = quaternion_to_matrix(L_arm_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_l_forearm_ori(ori_init):
    number = 0
    while True:
        L_forearm_quaternion = L_forearm[0]
        ori_init[8] = quaternion_to_matrix(L_forearm_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_back_ori(ori_init):
    number = 0
    while True:
        Back_quaternion = Back[0]
        ori_init[9] = quaternion_to_matrix(Back_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_r_foot_ori(ori_init):
    number = 0
    while True:
        R_foot_quaternion = R_foot[0]
        ori_init[10] = quaternion_to_matrix(R_foot_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_l_foot_ori(ori_init):
    number = 0
    while True:
        L_foot_quaternion = L_foot[0]
        ori_init[11] = quaternion_to_matrix(L_foot_quaternion)
        number += 1
        if number > 500:
            break
    return ori_init


def calibration_r_leg_y(ori_init):
    global last_angle_r_hip, last_angle_r_shank, last_angle_r_ankle
    number = 0
    r_hip_y = np.array([0, 0, 0])
    r_shank_y = np.array([0, 0, 0])
    r_ankle_y = np.array([0, 0, 0])
    while True:
        R_thigh_quaternion = R_thigh[0]
        r_thigh_ori = quaternion_to_matrix(R_thigh_quaternion)
        rot_hip = ori_init[1].T @ r_thigh_ori
        n_x1 = RotationMatrixToAxisAngle(rot_hip)[0]
        n_y1 = RotationMatrixToAxisAngle(rot_hip)[1]
        n_z1 = RotationMatrixToAxisAngle(rot_hip)[2]
        angle_hip = RotationMatrixToAxisAngle(rot_hip)[3]
        # print(angle_hip)
        R_shank_quaternion = R_shank[0]
        r_shank_ori = quaternion_to_matrix(R_shank_quaternion)
        rot_shank = ori_init[2].T @ r_shank_ori
        n_x2 = RotationMatrixToAxisAngle(rot_shank)[0]
        n_y2 = RotationMatrixToAxisAngle(rot_shank)[1]
        n_z2 = RotationMatrixToAxisAngle(rot_shank)[2]
        angle_shank = RotationMatrixToAxisAngle(rot_shank)[3]
        # print(angle_shank)
        R_foot_quaternion = R_foot[0]
        r_foot_ori = quaternion_to_matrix(R_foot_quaternion)
        rot_ankle = ori_init[10].T @ r_foot_ori
        n_x3 = RotationMatrixToAxisAngle(rot_ankle)[0]
        n_y3 = RotationMatrixToAxisAngle(rot_ankle)[1]
        n_z3 = RotationMatrixToAxisAngle(rot_ankle)[2]
        angle_ankle = RotationMatrixToAxisAngle(rot_ankle)[3]
        # print(angle_ankle)
        if abs(angle_hip - last_angle_r_hip) < 0.1 and abs(angle_shank - last_angle_r_shank) < 0.1 and \
                abs(angle_ankle - last_angle_r_ankle) < 0.1:
            r_hip_y = r_hip_y + np.array([abs(n_x1), abs(n_y1), abs(n_z1)])
            r_shank_y = r_shank_y + np.array([abs(n_x2), abs(n_y2), abs(n_z2)])
            r_ankle_y = r_ankle_y + np.array([abs(n_x3), abs(n_y3), abs(n_z3)])
            number += 1
        last_angle_r_hip = angle_hip
        last_angle_r_shank = angle_shank
        last_angle_r_ankle = angle_ankle
        if number == 1000:
            break
    r_hip_y = r_hip_y / 1000
    r_hip_y = r_hip_y / np.sqrt(np.dot(r_hip_y, r_hip_y))
    print(r_hip_y)
    r_shank_y = r_shank_y / 1000
    r_shank_y = r_shank_y / np.sqrt(np.dot(r_shank_y, r_shank_y))
    print(r_shank_y)
    r_ankle_y = r_ankle_y / 1000
    r_ankle_y = r_ankle_y / np.sqrt(np.dot(r_ankle_y, r_ankle_y))
    print(r_ankle_y)
    return r_hip_y, r_shank_y, r_ankle_y


def calibration_r_leg_z():
    number = 0
    while True:
        r_hip_z = R_thigh[2]
        r_hip_z = r_hip_z / np.sqrt(np.dot(r_hip_z, r_hip_z))

        r_shank_z = R_shank[2]
        r_shank_z = r_shank_z / np.sqrt(np.dot(r_shank_z, r_shank_z))

        r_ankle_z = R_foot[2]
        r_ankle_z = r_ankle_z / np.sqrt(np.dot(r_ankle_z, r_ankle_z))
        number += 1

        if number == 1000:
            break
    print(r_hip_z, r_shank_z, r_ankle_z)

    return r_hip_z, r_shank_z, r_ankle_z


def calibration_r_leg(ori_init):
    r_z = calibration_r_leg_z()
    r_hip_z = r_z[0]
    r_shank_z = r_z[1]
    r_ankle_z = r_z[2]
    print('右腿弯曲')
    time.sleep(5)
    r_y = calibration_r_leg_y(ori_init)
    r_hip_y = r_y[0]
    r_shank_y = r_y[1]
    r_ankle_y = r_y[2]

    r_hip_x = np.cross(r_hip_z, r_hip_y)
    r_hip_x = r_hip_x / np.sqrt(np.dot(r_hip_x, r_hip_x))
    r_hip_y = np.cross(r_hip_x, r_hip_z)
    r_hip_z = [-r_hip_z[0], -r_hip_z[1], -r_hip_z[2]]
    rot_r_hip = np.vstack((r_hip_x, r_hip_y, r_hip_z))
    print('rot_r_hip', rot_r_hip)
    r_shank_x = np.cross(r_shank_z, r_shank_y)
    r_shank_x = r_shank_x / np.sqrt(np.dot(r_shank_x, r_shank_x))
    r_shank_y = np.cross(r_shank_x, r_shank_z)
    r_shank_z = [-r_shank_z[0], -r_shank_z[1], -r_shank_z[2]]
    rot_r_shank = np.vstack((r_shank_x, r_shank_y, r_shank_z))
    print('rot_r_shank', rot_r_shank)
    r_ankle_x = np.cross(r_ankle_z, r_ankle_y)
    r_ankle_x = r_ankle_x / np.sqrt(np.dot(r_ankle_x, r_ankle_x))
    r_ankle_y = np.cross(r_ankle_x, r_ankle_z)
    r_ankle_z = [-r_ankle_z[0], -r_ankle_z[1], -r_ankle_z[2]]
    rot_r_ankle = np.vstack((r_ankle_x, r_ankle_y, r_ankle_z))
    print('rot_r_ankle', rot_r_ankle)
    return rot_r_hip, rot_r_shank, rot_r_ankle


def calibration_l_leg_y(ori_init):
    global last_angle_l_hip, last_angle_l_shank, last_angle_l_ankle
    number = 0
    l_hip_y = np.array([0, 0, 0])
    l_shank_y = np.array([0, 0, 0])
    l_ankle_y = np.array([0, 0, 0])
    while True:
        L_thigh_quaternion = L_thigh[0]
        l_thigh_ori = quaternion_to_matrix(L_thigh_quaternion)
        rot_hip = ori_init[5].T @ l_thigh_ori
        n_x1 = RotationMatrixToAxisAngle(rot_hip)[0]
        n_y1 = RotationMatrixToAxisAngle(rot_hip)[1]
        n_z1 = RotationMatrixToAxisAngle(rot_hip)[2]
        angle_hip = RotationMatrixToAxisAngle(rot_hip)[3]
        # print(angle_hip)
        L_shank_quaternion = L_shank[0]
        l_shank_ori = quaternion_to_matrix(L_shank_quaternion)
        rot_shank = ori_init[6].T @ l_shank_ori
        n_x2 = RotationMatrixToAxisAngle(rot_shank)[0]
        n_y2 = RotationMatrixToAxisAngle(rot_shank)[1]
        n_z2 = RotationMatrixToAxisAngle(rot_shank)[2]
        angle_shank = RotationMatrixToAxisAngle(rot_shank)[3]
        # print(angle_shank)
        L_foot_quaternion = L_foot[0]
        l_foot_ori = quaternion_to_matrix(L_foot_quaternion)
        rot_ankle = ori_init[11].T @ l_foot_ori
        n_x3 = RotationMatrixToAxisAngle(rot_ankle)[0]
        n_y3 = RotationMatrixToAxisAngle(rot_ankle)[1]
        n_z3 = RotationMatrixToAxisAngle(rot_ankle)[2]
        angle_ankle = RotationMatrixToAxisAngle(rot_ankle)[3]
        # print(angle_ankle)
        if abs(angle_hip - last_angle_l_hip) < 0.1 and abs(angle_shank - last_angle_l_shank) < 0.1 and \
                abs(angle_ankle - last_angle_l_ankle) < 0.1:
            l_hip_y = l_hip_y + np.array([abs(n_x1), abs(n_y1), abs(n_z1)])
            l_shank_y = l_shank_y + np.array([abs(n_x2), abs(n_y2), abs(n_z2)])
            l_ankle_y = l_ankle_y + np.array([abs(n_x3), abs(n_y3), abs(n_z3)])
            number += 1
        last_angle_l_hip = angle_hip
        last_angle_l_shank = angle_shank
        last_angle_l_ankle = angle_ankle
        if number == 1000:
            break
    l_hip_y = l_hip_y / 1000
    l_hip_y = l_hip_y / np.sqrt(np.dot(l_hip_y, l_hip_y))
    print(l_hip_y)
    l_shank_y = l_shank_y / 1000
    l_shank_y = l_shank_y / np.sqrt(np.dot(l_shank_y, l_shank_y))
    print(l_shank_y)
    l_ankle_y = l_ankle_y / 1000
    l_ankle_y = l_ankle_y / np.sqrt(np.dot(l_ankle_y, l_ankle_y))
    print(l_ankle_y)
    return l_hip_y, l_shank_y, l_ankle_y


def calibration_l_leg_z():
    number = 0
    while True:
        l_hip_z = L_thigh[2]
        l_hip_z = l_hip_z / np.sqrt(np.dot(l_hip_z, l_hip_z))

        l_shank_z = L_shank[2]
        l_shank_z = l_shank_z / np.sqrt(np.dot(l_shank_z, l_shank_z))

        l_ankle_z = L_foot[2]
        l_ankle_z = l_ankle_z / np.sqrt(np.dot(l_ankle_z, l_ankle_z))
        number += 1

        if number == 1000:
            break
    print(l_hip_z, l_shank_z, l_ankle_z)

    return l_hip_z, l_shank_z, l_ankle_z


def calibration_l_leg(ori_init):
    l_z = calibration_l_leg_z()
    l_hip_z = l_z[0]
    l_shank_z = l_z[1]
    l_ankle_z = l_z[2]
    print('左腿弯曲')
    time.sleep(5)
    l_y = calibration_l_leg_y(ori_init)
    l_hip_y = l_y[0]
    l_shank_y = l_y[1]
    l_ankle_y = l_y[2]

    l_hip_x = np.cross(l_hip_z, l_hip_y)
    l_hip_x = l_hip_x / np.sqrt(np.dot(l_hip_x, l_hip_x))
    l_hip_y = np.cross(l_hip_x, l_hip_z)
    l_hip_z = [-l_hip_z[0], -l_hip_z[1], -l_hip_z[2]]
    rot_l_hip = np.vstack((l_hip_x, l_hip_y, l_hip_z))
    print('rot_l_hip', rot_l_hip)
    l_shank_x = np.cross(l_shank_z, l_shank_y)
    l_shank_x = l_shank_x / np.sqrt(np.dot(l_shank_x, l_shank_x))
    l_shank_y = np.cross(l_shank_x, l_shank_z)
    l_shank_z = [-l_shank_z[0], -l_shank_z[1], -l_shank_z[2]]
    rot_l_shank = np.vstack((l_shank_x, l_shank_y, l_shank_z))
    print('rot_l_shank', rot_l_shank)
    l_ankle_x = np.cross(l_ankle_z, l_ankle_y)
    l_ankle_x = l_ankle_x / np.sqrt(np.dot(l_ankle_x, l_ankle_x))
    l_ankle_y = np.cross(l_ankle_x, l_ankle_z)
    l_ankle_z = [-l_ankle_z[0], -l_ankle_z[1], -l_ankle_z[2]]
    rot_l_ankle = np.vstack((l_ankle_x, l_ankle_y, l_ankle_z))
    print('rot_l_ankle', rot_l_ankle)
    return rot_l_hip, rot_l_shank, rot_l_ankle


def calibration_r_arm_y(ori_init):
    global last_angle_r_shoulder, last_angle_r_forearm
    number = 0
    r_shoulder_y = np.array([0, 0, 0])
    r_forearm_y = np.array([0, 0, 0])
    while True:
        R_arm_quaternion = R_arm[0]
        r_arm_ori = quaternion_to_matrix(R_arm_quaternion)
        rot_shoulder = ori_init[3].T @ r_arm_ori
        n_x1 = RotationMatrixToAxisAngle(rot_shoulder)[0]
        n_y1 = RotationMatrixToAxisAngle(rot_shoulder)[1]
        n_z1 = RotationMatrixToAxisAngle(rot_shoulder)[2]
        angle_shoulder = RotationMatrixToAxisAngle(rot_shoulder)[3]
        # print(angle_shoulder)
        R_forearm_quaternion = R_forearm[0]
        r_forearm_ori = quaternion_to_matrix(R_forearm_quaternion)
        rot_forearm = ori_init[4].T @ r_forearm_ori
        n_x2 = RotationMatrixToAxisAngle(rot_forearm)[0]
        n_y2 = RotationMatrixToAxisAngle(rot_forearm)[1]
        n_z2 = RotationMatrixToAxisAngle(rot_forearm)[2]
        angle_forearm = RotationMatrixToAxisAngle(rot_forearm)[3]
        # print(angle_forearm)
        if abs(angle_shoulder - last_angle_r_shoulder) < 0.1 and abs(angle_forearm - last_angle_r_forearm) < 0.1:
            r_shoulder_y = r_shoulder_y + np.array([abs(n_x1), abs(n_y1), abs(n_z1)])
            r_forearm_y = r_forearm_y + np.array([abs(n_x2), abs(n_y2), abs(n_z2)])
            number += 1
        last_angle_r_shoulder = angle_shoulder
        last_angle_r_forearm = angle_forearm
        if number == 1000:
            break
    r_shoulder_y = r_shoulder_y / 1000
    r_shoulder_y = r_shoulder_y / np.sqrt(np.dot(r_shoulder_y, r_shoulder_y))
    print(r_shoulder_y)
    r_forearm_y = r_forearm_y / 1000
    r_forearm_y = r_forearm_y / np.sqrt(np.dot(r_forearm_y, r_forearm_y))
    print(r_forearm_y)
    return r_shoulder_y, r_forearm_y


def calibration_r_arm_z():
    number = 0
    while True:
        r_shoulder_z = R_arm[2]
        r_shoulder_z = r_shoulder_z / np.sqrt(np.dot(r_shoulder_z, r_shoulder_z))

        r_forearm_z = R_forearm[2]
        r_forearm_z = r_forearm_z / np.sqrt(np.dot(r_forearm_z, r_forearm_z))

        number += 1
        if number == 500:
            break
    print(r_shoulder_z, r_forearm_z)

    return r_shoulder_z, r_forearm_z


def calibration_r_arm(ori_init):
    r_z = calibration_r_arm_z()
    r_shoulder_z = r_z[0]
    r_forearm_z = r_z[1]
    print('抬右臂')
    time.sleep(5)
    r_y = calibration_r_arm_y(ori_init)
    r_shoulder_y = r_y[0]
    r_forearm_y = r_y[1]

    r_shoulder_x = np.cross(r_shoulder_z, r_shoulder_y)
    r_shoulder_x = r_shoulder_x / np.sqrt(np.dot(r_shoulder_x, r_shoulder_x))
    r_shoulder_y = np.cross(r_shoulder_x, r_shoulder_z)
    r_shoulder_z = [-r_shoulder_z[0], -r_shoulder_z[1], -r_shoulder_z[2]]
    rot_r_shoulder = np.vstack((r_shoulder_x, r_shoulder_y, r_shoulder_z))
    print('rot_r_shoulder', rot_r_shoulder)
    r_forearm_x = np.cross(r_forearm_z, r_forearm_y)
    r_forearm_x = r_forearm_x / np.sqrt(np.dot(r_forearm_x, r_forearm_x))
    r_forearm_y = np.cross(r_forearm_x, r_forearm_z)
    r_forearm_z = [-r_forearm_z[0], -r_forearm_z[1], -r_forearm_z[2]]
    rot_r_forearm = np.vstack((r_forearm_x, r_forearm_y, r_forearm_z))
    print('rot_r_forearm', rot_r_forearm)
    return rot_r_shoulder, rot_r_forearm


def calibration_l_arm_y(ori_init):
    global last_angle_l_shoulder, last_angle_l_forearm
    number = 0
    l_shoulder_y = np.array([0, 0, 0])
    l_forearm_y = np.array([0, 0, 0])
    while True:
        L_arm_quaternion = L_arm[0]
        l_arm_ori = quaternion_to_matrix(L_arm_quaternion)
        rot_shoulder = ori_init[7].T @ l_arm_ori
        n_x1 = RotationMatrixToAxisAngle(rot_shoulder)[0]
        n_y1 = RotationMatrixToAxisAngle(rot_shoulder)[1]
        n_z1 = RotationMatrixToAxisAngle(rot_shoulder)[2]
        angle_shoulder = RotationMatrixToAxisAngle(rot_shoulder)[3]
        # print(angle_shoulder)
        L_forearm_quaternion = L_forearm[0]
        l_forearm_ori = quaternion_to_matrix(L_forearm_quaternion)
        rot_forearm = ori_init[8].T @ l_forearm_ori
        n_x2 = RotationMatrixToAxisAngle(rot_forearm)[0]
        n_y2 = RotationMatrixToAxisAngle(rot_forearm)[1]
        n_z2 = RotationMatrixToAxisAngle(rot_forearm)[2]
        angle_forearm = RotationMatrixToAxisAngle(rot_forearm)[3]
        # print(angle_forearm)
        if abs(angle_shoulder - last_angle_l_shoulder) < 0.1 and abs(angle_forearm - last_angle_l_forearm) < 0.1:
            l_shoulder_y = l_shoulder_y + np.array([abs(n_x1), abs(n_y1), abs(n_z1)])
            l_forearm_y = l_forearm_y + np.array([abs(n_x2), abs(n_y2), abs(n_z2)])
            number += 1
        last_angle_l_shoulder = angle_shoulder
        last_angle_l_forearm = angle_forearm
        if number == 1000:
            break
    l_shoulder_y = l_shoulder_y / 1000
    l_shoulder_y = l_shoulder_y / np.sqrt(np.dot(l_shoulder_y, l_shoulder_y))
    print(l_shoulder_y)
    l_forearm_y = l_forearm_y / 1000
    l_forearm_y = l_forearm_y / np.sqrt(np.dot(l_forearm_y, l_forearm_y))
    print(l_forearm_y)
    return l_shoulder_y, l_forearm_y


def calibration_l_arm_z():
    number = 0
    while True:
        l_shoulder_z = L_arm[2]
        l_shoulder_z = l_shoulder_z / np.sqrt(np.dot(l_shoulder_z, l_shoulder_z))

        l_forearm_z = L_forearm[2]
        l_forearm_z = l_forearm_z / np.sqrt(np.dot(l_forearm_z, l_forearm_z))

        number += 1
        if number == 500:
            break
    print(l_shoulder_z, l_forearm_z)

    return l_shoulder_z, l_forearm_z


# def calibration_l_arm(ori_init):
#     print('左臂弯曲')
#     time.sleep(5)
#     l_y = calibration_l_arm_y(ori_init)
#     l_shoulder_y = l_y[0]
#     l_forearm_y = l_y[1]
#     print('左臂外摆')
#     time.sleep(5)
#     l_x = calibration_l_arm_x(ori_init)
#     l_shoulder_x = l_x[0]
#     l_forearm_x = l_x[1]
#
#     l_shoulder_z = np.cross(l_shoulder_x, l_shoulder_y)
#     l_shoulder_z = l_shoulder_z / np.sqrt(np.dot(l_shoulder_z, l_shoulder_z))
#     rot_l_shoulder = np.vstack((l_shoulder_x, l_shoulder_y, l_shoulder_z))
#     print(rot_l_shoulder)
#     l_forearm_z = np.cross(l_forearm_x, l_forearm_y)
#     l_forearm_z = l_forearm_z / np.sqrt(np.dot(l_forearm_z, l_forearm_z))
#     rot_l_forearm = np.vstack((l_forearm_x, l_forearm_y, l_forearm_z))
#     print(rot_l_forearm)
#     return rot_l_shoulder, rot_l_forearm


def calibration_l_arm(ori_init):
    l_z = calibration_l_arm_z()
    l_shoulder_z = l_z[0]
    l_forearm_z = l_z[1]
    print('抬左臂')
    time.sleep(5)
    l_y = calibration_l_arm_y(ori_init)
    l_shoulder_y = l_y[0]
    l_forearm_y = l_y[1]

    l_shoulder_x = np.cross(l_shoulder_z, l_shoulder_y)
    l_shoulder_x = l_shoulder_x / np.sqrt(np.dot(l_shoulder_x, l_shoulder_x))
    l_shoulder_y = np.cross(l_shoulder_x, l_shoulder_z)
    l_shoulder_z = [-l_shoulder_z[0], -l_shoulder_z[1], -l_shoulder_z[2]]
    rot_l_shoulder = np.vstack((l_shoulder_x, l_shoulder_y, l_shoulder_z))
    print('rot_l_shoulder', rot_l_shoulder)
    l_forearm_x = np.cross(l_forearm_z, l_forearm_y)
    l_forearm_x = l_forearm_x / np.sqrt(np.dot(l_forearm_x, l_forearm_x))
    l_forearm_y = np.cross(l_forearm_x, l_forearm_z)
    l_forearm_z = [-l_forearm_z[0], -l_forearm_z[1], -l_forearm_z[2]]
    rot_l_forearm = np.vstack((l_forearm_x, l_forearm_y, l_forearm_z))
    print('rot_l_forearm', rot_l_forearm)
    return rot_l_shoulder, rot_l_forearm


def calibration_back_pelvis_y(ori_init):
    global last_angle_back, last_angle_pelvis
    number = 0
    back_y = np.array([0, 0, 0])
    pelvis_y = np.array([0, 0, 0])
    while True:
        Back_quaternion = Back[0]
        back_ori = quaternion_to_matrix(Back_quaternion)
        rot_back = ori_init[9].T @ back_ori
        n_x1 = RotationMatrixToAxisAngle(rot_back)[0]
        n_y1 = RotationMatrixToAxisAngle(rot_back)[1]
        n_z1 = RotationMatrixToAxisAngle(rot_back)[2]
        angle_back = RotationMatrixToAxisAngle(rot_back)[3]
        # print(angle_back)
        Pelvis_quaternion = Pelvis[0]
        pelvis_ori = quaternion_to_matrix(Pelvis_quaternion)
        rot_pelvis = ori_init[0].T @ pelvis_ori
        n_x2 = RotationMatrixToAxisAngle(rot_pelvis)[0]
        n_y2 = RotationMatrixToAxisAngle(rot_pelvis)[1]
        n_z2 = RotationMatrixToAxisAngle(rot_pelvis)[2]
        angle_pelvis = RotationMatrixToAxisAngle(rot_pelvis)[3]
        # print(angle_pelvis)
        if abs(angle_back - last_angle_back) < 0.1 and abs(angle_pelvis - last_angle_pelvis) < 0.1:
            back_y = back_y + np.array([abs(n_x1), abs(n_y1), abs(n_z1)])
            pelvis_y = pelvis_y + np.array([abs(n_x2), abs(n_y2), abs(n_z2)])
            number += 1
        last_angle_back = angle_back
        last_angle_pelvis = angle_pelvis
        if number == 1000:
            break
    back_y = back_y / 1000
    back_y = back_y / np.sqrt(np.dot(back_y, back_y))
    print(back_y)
    pelvis_y = pelvis_y / 1000
    pelvis_y = pelvis_y / np.sqrt(np.dot(pelvis_y, pelvis_y))
    print(pelvis_y)
    return back_y, pelvis_y


def calibration_back_pelvis_z():
    number = 0
    while True:
        back_z = Back[2]
        back_z = back_z / np.sqrt(np.dot(back_z, back_z))

        pelvis_z = Pelvis[2]
        pelvis_z = pelvis_z / np.sqrt(np.dot(pelvis_z, pelvis_z))

        number += 1
        if number == 1000:
            break
    print(back_z, pelvis_z)

    return back_z, pelvis_z


def calibration_back_pelvis(ori_init):
    z = calibration_back_pelvis_z()
    back_z = z[0]
    pelvis_z = z[1]
    print('后背弯曲')
    time.sleep(5)
    y = calibration_back_pelvis_y(ori_init)
    back_y = y[0]
    pelvis_y = y[1]

    back_x = np.cross(back_z, back_y)
    back_x = back_x / np.sqrt(np.dot(back_x, back_x))
    back_y = np.cross(back_x, back_z)
    back_z = [-back_z[0], -back_z[1], -back_z[2]]
    rot_back = np.vstack((back_x, back_y, back_z))
    print('rot_back', rot_back)
    pelvis_x = np.cross(pelvis_z, pelvis_y)
    pelvis_x = pelvis_x / np.sqrt(np.dot(pelvis_x, pelvis_x))
    pelvis_y = np.cross(pelvis_x, pelvis_z)
    pelvis_z = [-pelvis_z[0], -pelvis_z[1], -pelvis_z[2]]
    rot_pelvis = np.vstack((pelvis_x, pelvis_y, pelvis_z))
    print('rot_pelvis', rot_pelvis)
    return rot_pelvis, rot_pelvis


if __name__ == "__main__":
    ori_init = np.zeros((12, 3, 3))
    axis = []
    last_angle_r_hip = 0
    last_angle_r_shank = 0
    last_angle_r_ankle = 0
    last_angle_l_hip = 0
    last_angle_l_shank = 0
    last_angle_l_ankle = 0
    last_angle_r_shoulder = 0
    last_angle_r_forearm = 0
    last_angle_l_shoulder = 0
    last_angle_l_forearm = 0
    last_angle_back = 0
    last_angle_pelvis = 0

    Camera = [[]]
    R_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    R_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    R_arm = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    R_forearm = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    R_foot = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    L_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    L_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    L_arm = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    L_forearm = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    L_foot = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    Back = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    Pelvis = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    thread1 = threading.Thread(target=imu_data_485_usb, args=(L_thigh, L_shank, L_foot, L_arm, L_forearm, Back, '/dev/ttySC1', 6, ))
    thread1.start()
    thread2 = threading.Thread(target=imu_data_485_usb,args=(R_thigh, R_shank, R_foot, R_arm, R_forearm, Pelvis, '/dev/ttySC0', 6,))
    thread2.start()
    time.sleep(2)

    # R_thigh_quaternion = R_thigh[0]
    # R_shank_quaternion = R_shank[0]
    # R_foot_quaternion = R_foot[0]
    # R_arm_quaternion = R_arm[0]
    # R_forearm_quaternion = R_forearm[0]
    # L_thigh_quaternion = L_thigh[0]
    # L_shank_quaternion = L_shank[0]
    # L_foot_quaternion = L_foot[0]
    # L_arm_quaternion = L_arm[0]
    # L_forearm_quaternion = L_forearm[0]
    # Back_quaternion = Back[0]
    # Camera = Camera[0]

    calibration_r_thigh_ori(ori_init)
    calibration_r_shank_ori(ori_init)
    calibration_r_foot_ori(ori_init)
    calibration_r_arm_ori(ori_init)
    calibration_r_forearm_ori(ori_init)
    calibration_l_thigh_ori(ori_init)
    calibration_l_shank_ori(ori_init)
    calibration_l_foot_ori(ori_init)
    calibration_l_arm_ori(ori_init)
    calibration_l_forearm_ori(ori_init)
    calibration_back_ori(ori_init)
    calibration_pelvis_ori(ori_init)

    # # 右腿标定
    # time.sleep(3)
    # r_leg = calibration_r_leg(ori_init)
    # rot_r_hip = r_leg[0]
    # rot_r_shank = r_leg[1]
    # rot_r_ankle = r_leg[2]
    # np.savetxt(r'calibration/rot_r_hip.txt', rot_r_hip, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_r_shank.txt', rot_r_shank, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_r_ankle.txt', rot_r_ankle, fmt='%f', delimiter=',')

    # # 左腿标定
    # time.sleep(3)
    # l_leg = calibration_l_leg(ori_init)
    # rot_l_hip = l_leg[0]
    # rot_l_shank = l_leg[1]
    # rot_l_ankle = l_leg[2]
    # np.savetxt(r'calibration/rot_l_hip.txt', rot_l_hip, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_l_shank.txt', rot_l_shank, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_l_ankle.txt', rot_l_ankle, fmt='%f', delimiter=',')

    # # 右臂标定
    # time.sleep(3)
    # r_arm = calibration_r_arm(ori_init)
    # rot_r_shoulder = r_arm[0]
    # rot_r_forearm = r_arm[1]
    # np.savetxt(r'calibration/rot_r_shoulder.txt', rot_r_shoulder, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_r_forearm.txt', rot_r_forearm, fmt='%f', delimiter=',')

    # # 左臂标定
    # time.sleep(3)
    # l_arm = calibration_l_arm(ori_init)
    # rot_l_shoulder = l_arm[0]
    # rot_l_forearm = l_arm[1]
    # np.savetxt(r'calibration/rot_l_shoulder.txt', rot_l_shoulder, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_l_forearm.txt', rot_l_forearm, fmt='%f', delimiter=',')

    # # 后背骨盆标定
    # time.sleep(3)
    # back_pelvis = calibration_back_pelvis(ori_init)
    # rot_back = back_pelvis[0]
    # rot_pelvis = back_pelvis[1]
    # np.savetxt(r'calibration/rot_back.txt', rot_back, fmt='%f', delimiter=',')
    # np.savetxt(r'calibration/rot_pelvis.txt', rot_pelvis, fmt='%f', delimiter=',')

    # rot_test = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    # np.savetxt(r'calibration/rot_test.txt', rot_test, fmt='%f', delimiter=',')

    print('finish')
