import serial
import time
import struct
import zmq
import csv
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import linuxfd,signal,select
# from get_axis_position import get_axis_position
import datetime
import threading


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

rot_r_hip = np.loadtxt(r'calibration/rot_r_hip.txt', delimiter=',')
rot_l_hip = np.loadtxt(r'calibration/rot_l_hip.txt', delimiter=',')
rot_r_shank = np.loadtxt(r'calibration/rot_r_shank.txt', delimiter=',')
rot_l_shank = np.loadtxt(r'calibration/rot_l_shank.txt', delimiter=',')
rot_r_ankle = np.loadtxt(r'calibration/rot_r_ankle.txt', delimiter=',')
rot_l_ankle = np.loadtxt(r'calibration/rot_l_ankle.txt', delimiter=',')
rot_r_shoulder = np.loadtxt(r'calibration/rot_r_shoulder.txt', delimiter=',')
rot_l_shoulder = np.loadtxt(r'calibration/rot_l_shoulder.txt', delimiter=',')
rot_r_forearm = np.loadtxt(r'calibration/rot_r_forearm.txt', delimiter=',')
rot_l_forearm = np.loadtxt(r'calibration/rot_l_forearm.txt', delimiter=',')
rot_back = np.loadtxt(r'calibration/rot_back.txt', delimiter=',')
rot_pelvis = np.loadtxt(r'calibration/rot_pelvis.txt', delimiter=',')
rot_test = np.loadtxt(r'calibration/rot_test.txt', delimiter=',')

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


class imu_posture():
    def __init__(self):
        self.rot_r_hip = np.loadtxt(r'/home/pi/exoskeleton_master_zxt-v3.0/calibration/rot_r_hip.txt', delimiter=',')
        self.rot_l_hip = np.loadtxt(r'/home/pi/exoskeleton_master_zxt-v3.0/calibration/rot_l_hip.txt', delimiter=',')
        self.rot_r_shank = np.loadtxt(r'/home/pi/exoskeleton_master_zxt-v3.0/calibration/rot_r_shank.txt', delimiter=',')
        self.rot_l_shank = np.loadtxt(r'/home/pi/exoskeleton_master_zxt-v3.0/calibration/rot_l_shank.txt', delimiter=',')
        self.rot_pelvis = np.loadtxt(r'/home/pi/exoskeleton_master_zxt-v3.0/calibration/rot_pelvis.txt', delimiter=',')

        self.ser = serial.Serial('/dev/ttyUSB0', 921600, timeout=0.02)  # 115200,230400,460800
        # self.ser = serial.Serial('/dev/ttySC1', 921600, timeout=0.02)  # 115200,230400,460800
        self.txData = b'\xFF\xFF\x80\n'
        self.ser.write(self.txData)
        self.imu_num = 5             # imu数量
        self.imu_id = 0x80 + self.imu_num
        self.imu_data_dict = {}      #储存所有imu数据的字典
        self.imu_data_dict_right = {}
        self.imu_data_dict_left = {}
        for i in range(1, self.imu_num + 1):
            self.imu_data_dict_right[i - 1] = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
            self.imu_data_dict_left[i - 1] = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.model_file = '/home/pi/exo_hy/model/new/walk_new_quat_body0901.xml'
        self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
        self.data = mujoco.MjData(self.model)
        # print(self.data.qpos)
        self.r_thigh_ori_init = []
        self.r_shank_ori_init = []
        self.l_thigh_ori_init = []
        self.l_shank_ori_init = []
        self.pelvis_ori_init = []

        self.count = 0
        self.time_last = time.time()
        self.last_time = time.time()
        self.det_t = 0.02
        self.last_qvel = np.zeros(30)
        self.LIST = np.zeros(100)
        self.output = []
        self.position_dict = np.zeros(30)
        self.posture_dict = np.zeros(90)
        self.FINAL_data = np.zeros(159)
        self.FINAL_LIST = np.zeros(159)
        self.FINAL_output = np.zeros(159)

        self.R_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.R_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.L_thigh = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.L_shank = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.Pelvis = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def read_imu_data(self):
        readlen = 38 * self.imu_num  # 前16位四元数，17—22角速度，23-28线加速度a_raw，29-34线加速度a_real，35-38校验位：0xff,0xff,id,/n
        data = self.ser.read(readlen)
        l = len(data)
        if l == readlen:
            id = data[l - 2]
            if id == self.imu_id:
                for i in range(1, self.imu_num + 1):
                    start_index = (i - 1) * 34 + (i - 1) * 4
                    end_index = start_index + 34
                    result = struct.unpack('4f 3h 3h 3h', data[start_index: end_index])  # 解码
                    q_raw = list(result[0:4])
                    q = [q_raw[1], q_raw[2], q_raw[3], q_raw[0]]
                    g_raw = list(result[4:7])
                    g = [x / 938.734 for x in g_raw]
                    a_raw = list(result[7:10])
                    a_raw = ([a_raw[0] / 208.980, a_raw[1] / 208.980, a_raw[2] / 208.980])
                    a_real = list(result[10:13])
                    a_real = ([a_real[0] / 208.980, a_real[1] / 208.980, a_real[2] / 208.980])
                    Matrix = quaternion_to_matrix(q)
                    a_real = (a_raw - Matrix.T @ [0.0, 0.0, 9.8]).tolist()
                    imu_data = [q, g, a_raw, a_real]
                    self.imu_data_dict_left[i - 1] = imu_data  # 第k个imu的数据放在imu_data_dict的第k位
                    # print(imu_data)
                # print(self.imu_data_dict_left)
                # self.imu_data_dict_left[4] = self.Pelvis
                self.count = self.count + 1
            else:
                mujoco.mj_resetData(self.model, self.data)  # 重置数据
                print('wrong id =', id)
                print('data is :')
                for i in range(0, l):
                    print('%#x ' % data[i], end='')
                print('')

        else:
            mujoco.mj_resetData(self.model, self.data)  # 重置数据
            print('err: received length=', l)
            # print(self.data.qpos)

        self.ser.write(self.txData)
        return self.imu_data_dict_left

    # 初始标定
    def imu_calibration(self):
        # imu_data_list = [self.imu_data_dict_left[i] for i in range(0, self.imu_num)]
        imu_data_list = [self.imu_data_dict_left[i] for i in range(0, 5)]
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

    # 记录原始数据
    def get_rawdata(self):
        # imu_data_list = [self.imu_data_dict_left[i] for i in range(0, self.imu_num)]
        imu_data_list = [self.imu_data_dict_left[i] for i in range(0, 5)]
        self.R_shank, self.R_thigh, self.Pelvis, self.L_thigh, self.L_shank = imu_data_list

        # self.det_t = time.time() - self.last_time
        # self.last_time = time.time()
        self.det_t = 0.01
        # print(self.det_t)

        pelvis_angle_acc = [0, 0, 0]
        pelvis_angle_acc[0] = (Pelvis[1][0] - self.last_qvel[0]) / self.det_t
        pelvis_angle_acc[1] = (Pelvis[1][1] - self.last_qvel[1]) / self.det_t
        pelvis_angle_acc[2] = (Pelvis[1][2] - self.last_qvel[2]) / self.det_t
        self.last_qvel[0] = Pelvis[1][0]
        self.last_qvel[1] = Pelvis[1][1]
        self.last_qvel[2] = Pelvis[1][2]
        self.LIST[0] = Pelvis[3][0]
        self.LIST[1] = Pelvis[3][1]
        self.LIST[2] = Pelvis[3][2]
        self.LIST[3] = Pelvis[1][0]
        self.LIST[4] = Pelvis[1][1]
        self.LIST[5] = Pelvis[1][2]
        self.LIST[6] = pelvis_angle_acc[0]
        self.LIST[7] = pelvis_angle_acc[1]
        self.LIST[8] = pelvis_angle_acc[2]

        r_hip_angle_acc = [0, 0, 0]
        r_hip_angle_acc[0] = (R_thigh[1][0] - self.last_qvel[3]) / self.det_t
        r_hip_angle_acc[1] = (R_thigh[1][1] - self.last_qvel[4]) / self.det_t
        r_hip_angle_acc[2] = (R_thigh[1][2] - self.last_qvel[5]) / self.det_t
        self.last_qvel[3] = R_thigh[1][0]
        self.last_qvel[4] = R_thigh[1][1]
        self.last_qvel[5] = R_thigh[1][2]
        self.LIST[10] = R_thigh[3][0]
        self.LIST[11] = R_thigh[3][1]
        self.LIST[12] = R_thigh[3][2]
        self.LIST[13] = R_thigh[1][0]
        self.LIST[14] = R_thigh[1][1]
        self.LIST[15] = R_thigh[1][2]
        self.LIST[16] = r_hip_angle_acc[0]
        self.LIST[17] = r_hip_angle_acc[1]
        self.LIST[18] = r_hip_angle_acc[2]

        r_knee_angle_acc = [0, 0, 0]
        r_knee_angle_acc[0] = (R_shank[1][0] - self.last_qvel[6]) / self.det_t
        r_knee_angle_acc[1] = (R_shank[1][1] - self.last_qvel[7]) / self.det_t
        r_knee_angle_acc[2] = (R_shank[1][2] - self.last_qvel[8]) / self.det_t
        self.last_qvel[6] = R_shank[1][0]
        self.last_qvel[7] = R_shank[1][1]
        self.last_qvel[8] = R_shank[1][2]
        self.LIST[20] = R_shank[3][0]
        self.LIST[21] = R_shank[3][1]
        self.LIST[22] = R_shank[3][2]
        self.LIST[23] = R_shank[1][0]
        self.LIST[24] = R_shank[1][1]
        self.LIST[25] = R_shank[1][2]
        self.LIST[26] = r_knee_angle_acc[0]
        self.LIST[27] = r_knee_angle_acc[1]
        self.LIST[28] = r_knee_angle_acc[2]

        l_hip_angle_acc = [0, 0, 0]
        l_hip_angle_acc[0] = (L_thigh[1][0] - self.last_qvel[9]) / self.det_t
        l_hip_angle_acc[1] = (L_thigh[1][1] - self.last_qvel[10]) / self.det_t
        l_hip_angle_acc[2] = (L_thigh[1][2] - self.last_qvel[11]) / self.det_t
        self.last_qvel[9] = L_thigh[1][0]
        self.last_qvel[10] = L_thigh[1][1]
        self.last_qvel[11] = L_thigh[1][2]
        self.LIST[30] = L_thigh[3][0]
        self.LIST[31] = L_thigh[3][1]
        self.LIST[32] = L_thigh[3][2]
        self.LIST[33] = L_thigh[1][0]
        self.LIST[34] = L_thigh[1][1]
        self.LIST[35] = L_thigh[1][2]
        self.LIST[36] = l_hip_angle_acc[0]
        self.LIST[37] = l_hip_angle_acc[1]
        self.LIST[38] = l_hip_angle_acc[2]

        l_knee_angle_acc = [0, 0, 0]
        l_knee_angle_acc[0] = (L_shank[1][0] - self.last_qvel[12]) / self.det_t
        l_knee_angle_acc[1] = (L_shank[1][1] - self.last_qvel[13]) / self.det_t
        l_knee_angle_acc[2] = (L_shank[1][2] - self.last_qvel[14]) / self.det_t
        self.last_qvel[12] = L_shank[1][0]
        self.last_qvel[13] = L_shank[1][1]
        self.last_qvel[14] = L_shank[1][2]
        self.LIST[40] = L_shank[3][0]
        self.LIST[41] = L_shank[3][1]
        self.LIST[42] = L_shank[3][2]
        self.LIST[43] = L_shank[1][0]
        self.LIST[44] = L_shank[1][1]
        self.LIST[45] = L_shank[1][2]
        self.LIST[46] = l_knee_angle_acc[0]
        self.LIST[47] = l_knee_angle_acc[1]
        self.LIST[48] = l_knee_angle_acc[2]

        back_angle_acc = [0, 0, 0]
        back_angle_acc[0] = (Back[1][0] - self.last_qvel[15]) / self.det_t
        back_angle_acc[1] = (Back[1][1] - self.last_qvel[16]) / self.det_t
        back_angle_acc[2] = (Back[1][2] - self.last_qvel[17]) / self.det_t
        self.last_qvel[15] = Back[1][0]
        self.last_qvel[16] = Back[1][1]
        self.last_qvel[17] = Back[1][2]
        self.LIST[50] = Back[3][0]
        self.LIST[51] = Back[3][1]
        self.LIST[52] = Back[3][2]
        self.LIST[53] = Back[1][0]
        self.LIST[54] = Back[1][1]
        self.LIST[55] = Back[1][2]
        self.LIST[56] = back_angle_acc[0]
        self.LIST[57] = back_angle_acc[1]
        self.LIST[58] = back_angle_acc[2]

        r_shoulder_angle_acc = [0, 0, 0]
        r_shoulder_angle_acc[0] = (R_arm[1][0] - self.last_qvel[18]) / self.det_t
        r_shoulder_angle_acc[1] = (R_arm[1][1] - self.last_qvel[19]) / self.det_t
        r_shoulder_angle_acc[2] = (R_arm[1][2] - self.last_qvel[20]) / self.det_t
        self.last_qvel[18] = R_arm[1][0]
        self.last_qvel[19] = R_arm[1][1]
        self.last_qvel[20] = R_arm[1][2]
        self.LIST[60] = R_arm[3][0]
        self.LIST[61] = R_arm[3][1]
        self.LIST[62] = R_arm[3][2]
        self.LIST[63] = R_arm[1][0]
        self.LIST[64] = R_arm[1][1]
        self.LIST[65] = R_arm[1][2]
        self.LIST[66] = r_shoulder_angle_acc[0]
        self.LIST[67] = r_shoulder_angle_acc[1]
        self.LIST[68] = r_shoulder_angle_acc[2]

        r_elbow_angle_acc = [0, 0, 0]
        r_elbow_angle_acc[0] = (R_forearm[1][0] - self.last_qvel[21]) / self.det_t
        r_elbow_angle_acc[1] = (R_forearm[1][1] - self.last_qvel[22]) / self.det_t
        r_elbow_angle_acc[2] = (R_forearm[1][2] - self.last_qvel[23]) / self.det_t
        self.last_qvel[21] = R_forearm[1][0]
        self.last_qvel[22] = R_forearm[1][1]
        self.last_qvel[23] = R_forearm[1][2]
        self.LIST[70] = R_forearm[3][0]
        self.LIST[71] = R_forearm[3][1]
        self.LIST[72] = R_forearm[3][2]
        self.LIST[73] = R_forearm[1][0]
        self.LIST[74] = R_forearm[1][1]
        self.LIST[75] = R_forearm[1][2]
        self.LIST[76] = r_elbow_angle_acc[0]
        self.LIST[77] = r_elbow_angle_acc[1]
        self.LIST[78] = r_elbow_angle_acc[2]

        l_shoulder_angle_acc = [0, 0, 0]
        l_shoulder_angle_acc[0] = (L_arm[1][0] - self.last_qvel[24]) / self.det_t
        l_shoulder_angle_acc[1] = (L_arm[1][1] - self.last_qvel[25]) / self.det_t
        l_shoulder_angle_acc[2] = (L_arm[1][2] - self.last_qvel[26]) / self.det_t
        self.last_qvel[24] = L_arm[1][0]
        self.last_qvel[25] = L_arm[1][1]
        self.last_qvel[26] = L_arm[1][2]
        self.LIST[80] = L_arm[3][0]
        self.LIST[81] = L_arm[3][1]
        self.LIST[82] = L_arm[3][2]
        self.LIST[83] = L_arm[1][0]
        self.LIST[84] = L_arm[1][1]
        self.LIST[85] = L_arm[1][2]
        self.LIST[86] = l_shoulder_angle_acc[0]
        self.LIST[87] = l_shoulder_angle_acc[1]
        self.LIST[88] = l_shoulder_angle_acc[2]

        l_elbow_angle_acc = [0, 0, 0]
        l_elbow_angle_acc[0] = (L_forearm[1][0] - self.last_qvel[27]) / self.det_t
        l_elbow_angle_acc[1] = (L_forearm[1][1] - self.last_qvel[28]) / self.det_t
        l_elbow_angle_acc[2] = (L_forearm[1][2] - self.last_qvel[29]) / self.det_t
        self.last_qvel[27] = L_forearm[1][0]
        self.last_qvel[28] = L_forearm[1][1]
        self.last_qvel[29] = L_forearm[1][2]
        self.LIST[90] = L_forearm[3][0]
        self.LIST[91] = L_forearm[3][1]
        self.LIST[92] = L_forearm[3][2]
        self.LIST[93] = L_forearm[1][0]
        self.LIST[94] = L_forearm[1][1]
        self.LIST[95] = L_forearm[1][2]
        self.LIST[96] = l_elbow_angle_acc[0]
        self.LIST[97] = l_elbow_angle_acc[1]
        self.LIST[98] = l_elbow_angle_acc[2]

        self.LIST = np.array(self.LIST)
        # self.output.append(self.LIST)
        self.output = self.LIST

        self.FINAL_data[3] = self.data.qpos[3]
        self.FINAL_data[4] = self.data.qpos[4]
        self.FINAL_data[5] = self.data.qpos[5]
        self.FINAL_data[6] = self.data.qpos[6]  # 骨盆qpos
        self.FINAL_data[7] = self.data.qpos[7]
        self.FINAL_data[8] = self.data.qpos[8]
        self.FINAL_data[9] = self.data.qpos[9]
        self.FINAL_data[10] = self.data.qpos[10]  # 右髋qpos
        self.FINAL_data[11] = self.data.qpos[11]
        # self.FINAL_data[12] = self.data.qpos[12]
        # self.FINAL_data[13] = self.data.qpos[13]
        self.FINAL_data[14] = self.data.qpos[14]  # 右膝qpos
        self.FINAL_data[15] = self.data.qpos[15]
        self.FINAL_data[16] = self.data.qpos[16]
        self.FINAL_data[17] = self.data.qpos[17]
        self.FINAL_data[18] = self.data.qpos[18]  # 右踝qpos
        self.FINAL_data[21] = self.data.qpos[21]
        self.FINAL_data[22] = self.data.qpos[22]
        self.FINAL_data[23] = self.data.qpos[23]
        self.FINAL_data[24] = self.data.qpos[24]  # 左髋qpos
        self.FINAL_data[25] = self.data.qpos[25]
        # self.FINAL_data[26] = self.data.qpos[26]
        # self.FINAL_data[27] = self.data.qpos[27]
        self.FINAL_data[28] = self.data.qpos[28]  # 左膝qpos
        self.FINAL_data[29] = self.data.qpos[29]
        self.FINAL_data[30] = self.data.qpos[30]
        self.FINAL_data[31] = self.data.qpos[31]
        self.FINAL_data[32] = self.data.qpos[32]  # 左踝qpos
        self.FINAL_data[35] = self.data.qpos[35]
        self.FINAL_data[36] = self.data.qpos[36]
        self.FINAL_data[37] = self.data.qpos[37]
        self.FINAL_data[38] = self.data.qpos[38]  # 后背 qpos
        self.FINAL_data[39] = self.data.qpos[39]
        # self.FINAL_data[40] = self.data.qpos[40]
        # self.FINAL_data[41] = self.data.qpos[41]
        self.FINAL_data[42] = self.data.qpos[42]  # 右肩 qpos
        self.FINAL_data[43] = self.data.qpos[43]
        # self.FINAL_data[44] = self.data.qpos[44]
        # self.FINAL_data[45] = self.data.qpos[45]
        self.FINAL_data[46] = self.data.qpos[46]  # 右肘 qpos

        self.FINAL_data[47] = self.data.qpos[47]

        self.FINAL_data[50] = self.data.qpos[50]
        # self.FINAL_data[51] = self.data.qpos[51]
        # self.FINAL_data[52] = self.data.qpos[52]
        self.FINAL_data[53] = self.data.qpos[53]  # 右肩 qpos
        self.FINAL_data[54] = self.data.qpos[54]
        # self.FINAL_data[55] = self.data.qpos[55]
        # self.FINAL_data[56] = self.data.qpos[56]
        self.FINAL_data[57] = self.data.qpos[57]  # 右肘 qpos

        self.FINAL_data[58] = self.data.qpos[58]

        # qpos共61个数据  3平移+12ball*4+10hinge

        self.FINAL_data[61] = 0
        self.FINAL_data[62] = 0
        self.FINAL_data[63] = 0

        self.FINAL_data[64] = Pelvis[1][0]
        self.FINAL_data[65] = Pelvis[1][1]
        self.FINAL_data[66] = Pelvis[1][2]

        self.FINAL_data[67] = R_thigh[1][0]
        self.FINAL_data[68] = R_thigh[1][1]
        self.FINAL_data[69] = R_thigh[1][2]

        self.FINAL_data[70] = R_shank[1][0]
        self.FINAL_data[71] = R_shank[1][1]
        self.FINAL_data[72] = R_shank[1][2]

        self.FINAL_data[73] = R_foot[1][0]
        self.FINAL_data[74] = R_foot[1][1]
        self.FINAL_data[75] = R_foot[1][2]

        self.FINAL_data[78] = L_thigh[1][0]
        self.FINAL_data[79] = L_thigh[1][1]
        self.FINAL_data[80] = L_thigh[1][2]

        self.FINAL_data[81] = L_shank[1][0]
        self.FINAL_data[82] = L_shank[1][1]
        self.FINAL_data[83] = L_shank[1][2]

        self.FINAL_data[84] = L_foot[1][0]
        self.FINAL_data[85] = L_foot[1][1]
        self.FINAL_data[86] = L_foot[1][2]

        self.FINAL_data[89] = Back[1][0]
        self.FINAL_data[90] = Back[1][1]
        self.FINAL_data[91] = Back[1][2]

        self.FINAL_data[92] = R_arm[1][0]
        self.FINAL_data[93] = R_arm[1][1]
        self.FINAL_data[94] = R_arm[1][2]

        self.FINAL_data[95] = R_forearm[1][0]
        self.FINAL_data[96] = R_forearm[1][1]
        self.FINAL_data[97] = R_forearm[1][2]

        self.FINAL_data[101] = L_arm[1][0]
        self.FINAL_data[102] = L_arm[1][1]
        self.FINAL_data[103] = L_arm[1][2]

        self.FINAL_data[104] = L_forearm[1][0]
        self.FINAL_data[105] = L_forearm[1][1]
        self.FINAL_data[106] = L_forearm[1][2]
        # 107.108.109

        self.FINAL_data[110] = 0
        self.FINAL_data[111] = 0
        self.FINAL_data[112] = 0

        self.FINAL_data[113] = pelvis_angle_acc[0]  # Pelvis X-axis angular acceleration
        self.FINAL_data[114] = pelvis_angle_acc[1]  # Pelvis Y-axis angular acceleration
        self.FINAL_data[115] = pelvis_angle_acc[2]  # Pelvis Z-axis angular acceleration

        self.FINAL_data[116] = r_hip_angle_acc[0]  # Right thigh X-axis angular acceleration
        self.FINAL_data[117] = r_hip_angle_acc[1]  # Right thigh Y-axis angular acceleration
        self.FINAL_data[118] = r_hip_angle_acc[2]  # Right thigh Z-axis angular acceleration

        self.FINAL_data[119] = r_knee_angle_acc[0]  # Right shank X-axis angular acceleration
        self.FINAL_data[120] = r_knee_angle_acc[1]  # Right shank Y-axis angular acceleration
        self.FINAL_data[121] = r_knee_angle_acc[2]  # Right shank Z-axis angular acceleration

        # self.FINAL_data[122] = r_foot_angle_acc[0]  # Right foot X-axis angular acceleration
        # self.FINAL_data[123] = r_foot_angle_acc[1]  # Right foot Y-axis angular acceleration
        # self.FINAL_data[124] = r_foot_angle_acc[2]  # Right foot Z-axis angular acceleration

        self.FINAL_data[127] = l_hip_angle_acc[0]  # Left thigh X-axis angular acceleration
        self.FINAL_data[128] = l_hip_angle_acc[1]  # Left thigh Y-axis angular acceleration
        self.FINAL_data[129] = l_hip_angle_acc[2]  # Left thigh Z-axis angular acceleration

        self.FINAL_data[130] = l_knee_angle_acc[0]  # Left shank X-axis angular acceleration
        self.FINAL_data[131] = l_knee_angle_acc[1]  # Left shank Y-axis angular acceleration
        self.FINAL_data[132] = l_knee_angle_acc[2]  # Left shank Z-axis angular acceleration

        # self.FINAL_data[133] = l_foot_angle_acc[0]  # Duplicate of left foot X
        # self.FINAL_data[134] = l_foot_angle_acc[1]  # Duplicate of left foot Y
        # self.FINAL_data[135] = l_foot_angle_acc[2]  # Duplicate of left foot Z

        self.FINAL_data[138] = back_angle_acc[0]  # Back X-axis angular acceleration
        self.FINAL_data[139] = back_angle_acc[1]  # Back Y-axis angular acceleration
        self.FINAL_data[140] = back_angle_acc[2]  # Back Z-axis angular acceleration

        self.FINAL_data[141] = r_shoulder_angle_acc[0]  # Right arm X-axis angular acceleration
        self.FINAL_data[142] = r_shoulder_angle_acc[1]  # Right arm Y-axis angular acceleration
        self.FINAL_data[143] = r_shoulder_angle_acc[2]  # Right arm Z-axis angular acceleration

        self.FINAL_data[144] = r_elbow_angle_acc[0]  # Right forearm X-axis angular acceleration
        self.FINAL_data[145] = r_elbow_angle_acc[1]  # Right forearm Y-axis angular acceleration
        self.FINAL_data[146] = r_elbow_angle_acc[2]  # Right forearm Z-axis angular acceleration

        self.FINAL_data[150] = l_shoulder_angle_acc[0]  # Left arm X-axis angular acceleration
        self.FINAL_data[151] = l_shoulder_angle_acc[1]  # Left arm Y-axis angular acceleration
        self.FINAL_data[152] = l_shoulder_angle_acc[2]  # Left arm Z-axis angular acceleration

        self.FINAL_data[153] = l_elbow_angle_acc[0]  # Left forearm X-axis angular acceleration
        self.FINAL_data[154] = l_elbow_angle_acc[1]  # Left forearm Y-axis angular acceleration
        self.FINAL_data[155] = l_elbow_angle_acc[2]  # Left forearm Z-axis angular acceleration

        self.FINAL_LIST = np.array(self.FINAL_data)
        # self.output.append(self.LIST)
        self.FINAL_output = self.FINAL_LIST

    # imu数据解算
    def imu_data_solving(self):
        # imu_data_list = [self.imu_data_dict_left[i] for i in range(0, self.imu_num)]
        imu_data_list = [self.imu_data_dict_left[i] for i in range(0, 5)]
        self.R_shank, self.R_thigh, self.Pelvis, self.L_thigh, self.L_shank = imu_data_list

        '''骨盆位置'''   # X向前，Y向左，Z向上
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
        # data.qpos[12] = r_knee[0]
        # data.qpos[13] = r_knee[1]
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
        # data.qpos[26] = l_knee[0]
        # data.qpos[27] = l_knee[1]
        self.data.qpos[28] = -abs(l_knee[2])  # 弯曲

        mujoco.mj_forward(self.model, self.data)

    def save_data(self):
        with open('./output/data' + str(datetime.datetime.now().strftime('%Y年%m月%d日%H时%M分%S秒%f')[:-3]) + '.csv', 'w',
                  newline='') as file:
            # with open('./data' + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.output)



if __name__ == '__main__':
    print('process init')
    read_imu = imu_posture()

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:6655")
    print('process start')

    # while True:
    #     read_imu.read_imu_data_right()
    #     read_imu.read_imu_data_left()
    #     if read_imu.count < 300:
    #         read_imu.imu_calibration()  # 初始化
    #     elif read_imu.count == 300:
    #         print('calibration')
    #     # elif 300 < read_imu.count < 1000:
    #     #     read_imu.get_rawdata()  # 记录数据
    #     # elif read_imu.count == 1000:
    #     #     read_imu.calibration_position()
    #     #     print('start')
    #     #     # threading.Thread(target=read_imu.calibration_position).start()
    #     else:
    #         read_imu.imu_data_solving()  # 正常工作
    #         publisher.send_multipart([read_imu.data.qpos.tobytes(), read_imu.position_dict.tobytes(), read_imu.posture_dict.tobytes()])

    with open('./output/data' + str(datetime.datetime.now().strftime('%Y年%m月%d日%H时%M分%S秒%f')[:-3]) + '.csv', 'w',
              newline='') as file:


        # # 定时器
        # create special file objects
        efd = linuxfd.eventfd(initval=0, nonBlocking=True)
        sfd = linuxfd.signalfd(signalset={signal.SIGINT}, nonBlocking=True)
        tfd = linuxfd.timerfd(rtc=True, nonBlocking=True)
        # program timer and mask SIGINT
        tfd.settime(1, 0.01)  # 第一次定时器间隔和以后所有定时器间隔
        signal.pthread_sigmask(signal.SIG_SETMASK, {signal.SIGINT})
        # create epoll instance and register special files
        epl = select.epoll()
        epl.register(efd.fileno(), select.EPOLLIN)
        epl.register(sfd.fileno(), select.EPOLLIN)
        epl.register(tfd.fileno(), select.EPOLLIN)
        # start main loop
        isrunning = True
        while isrunning:
        #     block until epoll detects changes in the registered files
            events = epl.poll(-1)
            t = time.time()
        #     iterate over occurred events
            for fd, event in events:
                if fd == efd.fileno() and event & select.EPOLLIN:
        #             event file descriptor readable: read and exit loop
                    print("{0:.3f}: event file received update, exiting...".format(t))
                    efd.read()
                    isrunning = False
                elif fd == sfd.fileno() and event & select.EPOLLIN:
        #             signal file descriptor readable: write to event file
                    siginfo = sfd.read()
                    if siginfo["signo"] == signal.SIGINT:
        #                print("{0:.3f}: SIGINT received, notifying event file".format(t))
                       efd.write(1)
                elif fd == tfd.fileno() and event & select.EPOLLIN:
        #             timer file descriptor readable: display that timer has expired
                    tfd.read()
                    read_imu.read_imu_data()
                    if read_imu.count < 300:
                        read_imu.imu_calibration()  # 初始化
                    elif read_imu.count == 300:
                        print('calibration')
                    elif 300 < read_imu.count < 1000:
                        read_imu.get_rawdata()  # 记录数据
                    elif read_imu.count == 1000:
                        # read_imu.calibration_position()
                        print('start')
                        # threading.Thread(target=read_imu.calibration_position).start()
                    else:
                        read_imu.get_rawdata()  # 记录数据
                        # with open('./data' + '.csv', 'w', newline='') as file:
                        read_imu.imu_data_solving()  # 正常工作
                        writer = csv.writer(file)
                        writer.writerow(read_imu.FINAL_output)
                        # read_imu.save_data()
                        # read_imu.imu_data_solving()  # 正常工作
                        # publisher.send_multipart([read_imu.data.qpos.tobytes(), read_imu.position_dict.tobytes(), read_imu.posture_dict.tobytes()])
                        publisher.send_multipart([read_imu.data.qpos.tobytes(), read_imu.position_dict.tobytes(), read_imu.posture_dict.tobytes()])





