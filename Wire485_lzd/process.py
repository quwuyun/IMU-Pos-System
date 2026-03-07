import serial
import time
import struct
import zmq
import csv
import numpy as np
import mujoco       # pip install mujoco==2.3.7    pip install mujoco-python-viewer
from scipy.spatial.transform import Rotation
# import linuxfd,signal,select
from get_axis_position import get_axis_position
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
    quaternion = r.as_quat(canonical=False)
    return quaternion


class imu_posture:
    def __init__(self):
        self.ser_right = serial.Serial('/dev/ttySC0', 921600, timeout=0.01)#115200,230400,460800
        self.ser_left = serial.Serial('/dev/ttySC1', 921600, timeout=0.01)  # 115200,230400,460800
        self.txData = b'\xFF\xFF\x80\n'
        self.ser_right.write(self.txData)
        self.ser_left.write(self.txData)
        self.imu_num = 6             # imu数量
        self.imu_id = 0x80 + self.imu_num
        self.imu_data_dict = {}      #储存所有imu数据的字典
        self.imu_data_dict_right = {}
        self.imu_data_dict_left = {}
        for i in range(1, self.imu_num + 1):
            self.imu_data_dict_right[i - 1] = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
            self.imu_data_dict_left[i - 1] = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

        self.model_file = './model/new/walk_new_quat_body0901.xml'
        self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
        self.data = mujoco.MjData(self.model)

        self.r_thigh_ori_init = []
        self.r_shank_ori_init = []
        self.r_foot_ori_init = []
        self.r_arm_ori_init = []
        self.r_forearm_ori_init = []
        self.l_thigh_ori_init = []
        self.l_shank_ori_init = []
        self.l_foot_ori_init = []
        self.l_arm_ori_init = []
        self.l_forearm_ori_init =[]
        self.back_ori_init = []
        self.pelvis_ori_init = []

        self.count = 0
        self.time_last = time.time()
        self.last_time = time.time()
        self.det_t = 0.01
        self.last_qvel = np.zeros(30)
        self.LIST = np.zeros(100)
        self.output = []
        self.position_dict = np.zeros(30)
        self.posture_dict = np.zeros(90)

    def read_imu_data_right(self):
        readlen = 38 * self.imu_num  # 前16位四元数，17—22角速度，23-28线加速度a_raw，29-34线加速度a_real，35-38校验位：0xff,0xff,id,/n
        data = self.ser_right.read(readlen)
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
                    # a_real = list(result[10:13])
                    # a_real = ([a_real[0] / 208.980, a_real[1] / 208.980, a_real[2] / 208.980])
                    Matrix = quaternion_to_matrix(q)
                    a_real = (a_raw - Matrix.T @ [0.0, 0.0, 9.8]).tolist()
                    imu_data = [q, g, a_raw, a_real]
                    self.imu_data_dict_right[i - 1] = imu_data  # 第k个imu的数据放在imu_data_dict的第k位
                    # print(imu_data)
                # print(self.imu_data_dict_right)
                self.count = self.count + 1
            else:
                print('wrong id =', id)
                print('data is :')
                for i in range(0, l):
                    print('%#x ' % data[i], end='')
                print('')

            if self.count % 100 == 0:
                print('count =', self.count)
                print('time =', time.time() - self.time_last)
                self.time_last = time.time()

        else:
            print('err: received length=', l)

        self.ser_right.write(self.txData)
        return self.imu_data_dict_right

    def read_imu_data_left(self):
        readlen = 38 * self.imu_num  # 前16位四元数，17—22角速度，23-28线加速度a_raw，29-34线加速度a_real，35-38校验位：0xff,0xff,id,/n
        data = self.ser_left.read(readlen)
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
                    # a_real = list(result[10:13])
                    # a_real = ([a_real[0] / 208.980, a_real[1] / 208.980, a_real[2] / 208.980])
                    Matrix = quaternion_to_matrix(q)
                    a_real = (a_raw - Matrix.T @ [0.0, 0.0, 9.8]).tolist()
                    imu_data = [q, g, a_raw, a_real]
                    self.imu_data_dict_left[i - 1] = imu_data  # 第k个imu的数据放在imu_data_dict的第k位
                    # print(imu_data)
                # print(self.imu_data_dict_left)
                # self.count = self.count + 1
            else:
                print('wrong id =', id)
                print('data is :')
                for i in range(0, l):
                    print('%#x ' % data[i], end='')
                print('')

            # if self.count % 100 == 0:
            #     print('count =', self.count)
            #     print('time =', time.time() - self.time_last)
            #     self.time_last = time.time()

        else:
            print('err: received length=', l)

        self.ser_left.write(self.txData)
        return self.imu_data_dict_left

    # 初始标定
    def imu_calibration(self):
        imu_data_list_right = [self.imu_data_dict_right[i] for i in range(0, self.imu_num)]
        imu_data_list_left = [self.imu_data_dict_left[i] for i in range(0, self.imu_num)]
        R_thigh, R_shank, R_foot, R_arm, R_forearm, Pelvis = imu_data_list_right
        L_thigh, L_shank, L_foot, L_arm, L_forearm, Back = imu_data_list_left

        r_thigh_quaternion_init = R_thigh[0]
        r_shank_quaternion_init = R_shank[0]
        r_foot_quaternion_init = R_foot[0]
        r_arm_quaternion_init = R_arm[0]
        r_forearm_quaternion_init = R_forearm[0]
        l_thigh_quaternion_init = L_thigh[0]
        l_shank_quaternion_init = L_shank[0]
        l_foot_quaternion_init = L_foot[0]
        l_arm_quaternion_init = L_arm[0]
        l_forearm_quaternion_init = L_forearm[0]
        back_quaternion_init = Back[0]
        pelvis_quaternion_init = Pelvis[0]

        self.r_thigh_ori_init = quaternion_to_matrix(r_thigh_quaternion_init)
        self.r_shank_ori_init = quaternion_to_matrix(r_shank_quaternion_init)
        self.r_foot_ori_init = quaternion_to_matrix(r_foot_quaternion_init)
        self.r_arm_ori_init = quaternion_to_matrix(r_arm_quaternion_init)
        self.r_forearm_ori_init = quaternion_to_matrix(r_forearm_quaternion_init)
        self.l_thigh_ori_init = quaternion_to_matrix(l_thigh_quaternion_init)
        self.l_shank_ori_init = quaternion_to_matrix(l_shank_quaternion_init)
        self.l_foot_ori_init = quaternion_to_matrix(l_foot_quaternion_init)
        self.l_arm_ori_init = quaternion_to_matrix(l_arm_quaternion_init)
        self.l_forearm_ori_init = quaternion_to_matrix(l_forearm_quaternion_init)
        self.back_ori_init = quaternion_to_matrix(back_quaternion_init)
        self.pelvis_ori_init = quaternion_to_matrix(pelvis_quaternion_init)

    # 记录原始数据
    def get_rawdata(self):
        imu_data_list_right = [self.imu_data_dict_right[i] for i in range(0, self.imu_num)]
        imu_data_list_left = [self.imu_data_dict_left[i] for i in range(0, self.imu_num)]
        R_thigh, R_shank, R_foot, R_arm, R_forearm, Pelvis = imu_data_list_right
        L_thigh, L_shank, L_foot, L_arm, L_forearm, Back = imu_data_list_left

        self.det_t = time.time() - self.last_time
        self.last_time = time.time()
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
        self.output.append(self.LIST)

    # 计算imu位置、姿态
    def calibration_position(self):
        with open('./output/data' + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.output)
        print('finish!')

        axis_position = get_axis_position('data')
        position = axis_position.calculate_position()

        r_thigh_imu_position = position[0]
        r_shank_imu_position = position[1]
        l_thigh_imu_position = position[2]
        l_shank_imu_position = position[3]
        r_arm_imu_position = position[4]
        r_forearm_imu_position = position[5]
        l_arm_imu_position = position[6]
        l_forearm_imu_position = position[7]
        pelvis_imu_position = position[8]
        back_imu_position = position[9]

        r_thigh_imu_position = rot_r_hip @ r_thigh_imu_position
        r_shank_imu_position = rot_r_shank @ r_shank_imu_position
        l_thigh_imu_position = rot_l_hip @ l_thigh_imu_position
        l_shank_imu_position = rot_l_shank @ l_shank_imu_position
        r_arm_imu_position = rot_r_shoulder @ r_arm_imu_position
        r_forearm_imu_position = rot_r_forearm @ r_forearm_imu_position
        l_arm_imu_position = rot_l_shoulder @ l_arm_imu_position
        l_forearm_imu_position = rot_l_forearm @ l_forearm_imu_position
        pelvis_imu_position = rot_pelvis @ pelvis_imu_position
        back_imu_position = rot_back @ back_imu_position

        self.position_dict[0:3] = r_thigh_imu_position
        self.position_dict[3:6] = r_shank_imu_position
        self.position_dict[6:9] = l_thigh_imu_position
        self.position_dict[9:12] = l_shank_imu_position
        self.position_dict[12:15] = r_arm_imu_position
        self.position_dict[15:18] = r_forearm_imu_position
        self.position_dict[18:21] = l_arm_imu_position
        self.position_dict[21:24] = l_forearm_imu_position
        self.position_dict[24:27] = pelvis_imu_position
        self.position_dict[27:30] = back_imu_position

        self.position_dict = np.array(self.position_dict)
        self.position_dict = self.position_dict.reshape(30)
        # print(self.position_dict)

        r_thigh_imu_posture = (rot_test @ rot_r_hip.T).reshape(9)
        r_shank_imu_posture = (rot_test @ rot_r_shank.T).reshape(9)
        l_thigh_imu_posture = (rot_test @ rot_l_hip.T).reshape(9)
        l_shank_imu_posture = (rot_test @ rot_l_shank.T).reshape(9)
        r_arm_imu_posture = (rot_test @ rot_r_shoulder.T).reshape(9)
        r_forearm_imu_posture = (rot_test @ rot_r_forearm.T).reshape(9)
        l_arm_imu_posture = (rot_test @ rot_l_shoulder.T).reshape(9)
        l_forearm_imu_posture = (rot_test @ rot_l_forearm.T).reshape(9)
        pelvis_imu_posture = (rot_test @ rot_pelvis.T).reshape(9)
        back_imu_posture = (rot_test @ rot_back.T).reshape(9)

        self.posture_dict[0:9] = r_thigh_imu_posture
        self.posture_dict[9:18] = r_shank_imu_posture
        self.posture_dict[18:27] = l_thigh_imu_posture
        self.posture_dict[27:36] = l_shank_imu_posture
        self.posture_dict[36:45] = r_arm_imu_posture
        self.posture_dict[45:54] = r_forearm_imu_posture
        self.posture_dict[54:63] = l_arm_imu_posture
        self.posture_dict[63:72] = l_forearm_imu_posture
        self.posture_dict[72:81] = pelvis_imu_posture
        self.posture_dict[81:90] = back_imu_posture

        self.posture_dict = np.array(self.posture_dict)
        self.posture_dict = self.posture_dict.reshape(90)
        # print(self.posture_dict)

    # imu数据解算
    def imu_data_solving(self):
        imu_data_list_right = [self.imu_data_dict_right[i] for i in range(0, self.imu_num)]
        imu_data_list_left = [self.imu_data_dict_left[i] for i in range(0, self.imu_num)]
        R_thigh, R_shank, R_foot, R_arm, R_forearm, Pelvis = imu_data_list_right
        L_thigh, L_shank, L_foot, L_arm, L_forearm, Back = imu_data_list_left

        '''骨盆位置'''   # X向前，Y向左，Z向上
        # Position = camera[0]
        # data.qpos[0] = -Position.z
        # data.qpos[1] = -Position.x
        # data.qpos[2] = Position.y
        pelvis_quaternion = Pelvis[0]
        pelvis_ori = quaternion_to_matrix(pelvis_quaternion)
        Pelvis_ori = self.pelvis_ori_init.T @ pelvis_ori
        pelvis_quat = matrix_to_quaternion(Pelvis_ori)
        PELVIS_quat = rot_pelvis @ [pelvis_quat[0], pelvis_quat[1], pelvis_quat[2]]
        self.data.qpos[3] = pelvis_quat[3]
        self.data.qpos[4] = -PELVIS_quat[0]  # 外摆
        self.data.qpos[5] = -PELVIS_quat[2]
        self.data.qpos[6] = -PELVIS_quat[1]  # 弯曲
        T_pelvis = quaternion_to_matrix([-PELVIS_quat[0], -PELVIS_quat[2], -PELVIS_quat[1], pelvis_quat[3]]) 
        '''角速度'''
        pelvis_angle_velocity = rot_pelvis @ Pelvis[1]
        qvel_pelvis = pelvis_angle_velocity
        self.data.qvel[3] = -qvel_pelvis[0]
        self.data.qvel[4] = -qvel_pelvis[2]
        self.data.qvel[5] = -qvel_pelvis[1]

        '''右髋关节角度'''
        r_thigh_quaternion = R_thigh[0]
        r_thigh_ori = quaternion_to_matrix(r_thigh_quaternion)
        R_thigh_ori = self.r_thigh_ori_init.T @ r_thigh_ori
        r_thigh_quat = matrix_to_quaternion(R_thigh_ori)
        r_hip_quat = rot_r_hip @ [r_thigh_quat[0], r_thigh_quat[1], r_thigh_quat[2]]
        T_r_thigh = quaternion_to_matrix([-r_hip_quat[0], -r_hip_quat[2], -r_hip_quat[1], r_thigh_quat[3]])
        T_r_thigh_pelvis = T_pelvis.T @ T_r_thigh
        r_hip = matrix_to_quaternion(T_r_thigh_pelvis)
        self.data.qpos[7] = r_hip[3]
        self.data.qpos[8] = r_hip[0]  # 外摆
        self.data.qpos[9] = r_hip[1]
        self.data.qpos[10] = r_hip[2]  # 弯曲
        '''角速度'''
        r_thigh_angle_velocity = rot_r_hip @ R_thigh[1]
        qvel_r_hip = r_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[6] = -qvel_r_hip[0]
        self.data.qvel[7] = -qvel_r_hip[2]
        self.data.qvel[8] = -qvel_r_hip[1]

        '''右膝关节角度'''
        r_shank_quaternion = R_shank[0]
        r_shank_ori = quaternion_to_matrix(r_shank_quaternion)
        R_shank_ori = self.r_shank_ori_init.T @ r_shank_ori
        r_shank_quat = matrix_to_quaternion(R_shank_ori)
        r_knee_quat = rot_r_shank @ [r_shank_quat[0], r_shank_quat[1], r_shank_quat[2]]
        T_r_shank = quaternion_to_matrix([-r_knee_quat[0], -r_knee_quat[2], -r_knee_quat[1], r_shank_quat[3]])
        T_r_shank_thigh = T_r_thigh.T @ T_r_shank
        r_knee = matrix_to_quaternion(T_r_shank_thigh)
        self.data.qpos[11] = abs(r_knee[3])
        # data.qpos[12] = r_knee[0]
        # data.qpos[13] = r_knee[1]
        self.data.qpos[14] = -abs(r_knee[2])  # 弯曲
        '''角速度'''
        r_shank_angle_velocity = rot_r_shank @ R_shank[1]
        qvel_r_knee = r_shank_angle_velocity[1] - r_thigh_angle_velocity[1]
        self.data.qvel[11] = -qvel_r_knee

        '''右踝关节角度'''
        r_foot_quaternion = R_foot[0]
        r_foot_ori = quaternion_to_matrix(r_foot_quaternion)
        R_foot_ori = self.r_foot_ori_init.T @ r_foot_ori
        r_foot_quat = matrix_to_quaternion(R_foot_ori)
        r_ankle_quat = rot_r_ankle @ [r_foot_quat[0], r_foot_quat[1], r_foot_quat[2]]
        T_r_foot = quaternion_to_matrix([-r_ankle_quat[0], -r_ankle_quat[2], -r_ankle_quat[1], r_foot_quat[3]])
        T_r_foot_shank = T_r_shank.T @ T_r_foot
        r_ankle = matrix_to_quaternion(T_r_foot_shank)
        self.data.qpos[15] = r_ankle[3]
        self.data.qpos[16] = r_ankle[0]  # 外摆
        self.data.qpos[17] = r_ankle[1]
        self.data.qpos[18] = r_ankle[2]  # 弯曲
        '''角速度'''
        r_foot_angle_velocity = rot_r_ankle @ R_foot[1]
        qvel_r_ankle = r_foot_angle_velocity - r_shank_angle_velocity
        self.data.qvel[12] = -qvel_r_ankle[0]
        self.data.qvel[13] = -qvel_r_ankle[2]
        self.data.qvel[14] = -qvel_r_ankle[1]

        '''左髋关节角度'''
        l_thigh_quaternion = L_thigh[0]
        l_thigh_ori = quaternion_to_matrix(l_thigh_quaternion)
        L_thigh_ori = self.l_thigh_ori_init.T @ l_thigh_ori
        l_thigh_quat = matrix_to_quaternion(L_thigh_ori)
        l_hip_quat = rot_l_hip @ [l_thigh_quat[0], l_thigh_quat[1], l_thigh_quat[2]]
        T_l_thigh = quaternion_to_matrix([-l_hip_quat[0], -l_hip_quat[2], -l_hip_quat[1], l_thigh_quat[3]])
        T_l_thigh_pelvis = T_pelvis.T @ T_l_thigh
        l_hip = matrix_to_quaternion(T_l_thigh_pelvis)
        self.data.qpos[21] = l_hip[3]
        self.data.qpos[22] = l_hip[0]  # 外摆
        self.data.qpos[23] = l_hip[1]
        self.data.qpos[24] = l_hip[2]  # 弯曲
        '''角速度'''
        l_thigh_angle_velocity = rot_l_hip @ L_thigh[1]
        qvel_l_hip = l_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[17] = -qvel_l_hip[0]
        self.data.qvel[18] = -qvel_l_hip[2]
        self.data.qvel[19] = -qvel_l_hip[1]

        '''左膝关节角度'''
        l_shank_quaternion = L_shank[0]
        l_shank_ori = quaternion_to_matrix(l_shank_quaternion)
        L_shank_ori = self.l_shank_ori_init.T @ l_shank_ori
        l_shank_quat = matrix_to_quaternion(L_shank_ori)
        l_knee_quat = rot_l_shank @ [l_shank_quat[0], l_shank_quat[1], l_shank_quat[2]]
        T_l_shank = quaternion_to_matrix([-l_knee_quat[0], -l_knee_quat[2], -l_knee_quat[1], l_shank_quat[3]])
        T_l_shank_thigh = T_l_thigh.T @ T_l_shank
        l_knee = matrix_to_quaternion(T_l_shank_thigh)
        self.data.qpos[25] = abs(l_knee[3])
        # data.qpos[26] = l_knee[0]
        # data.qpos[27] = l_knee[1]
        self.data.qpos[28] = -abs(l_knee[2])  # 弯曲
        '''角速度'''
        l_shank_angle_velocity = rot_l_shank @ L_shank[1]
        qvel_l_knee = l_shank_angle_velocity[1] - l_thigh_angle_velocity[1]
        self.data.qvel[22] = -qvel_l_knee

        '''左踝关节角度'''
        l_foot_quaternion = L_foot[0]
        l_foot_ori = quaternion_to_matrix(l_foot_quaternion)
        L_foot_ori = self.l_foot_ori_init.T @ l_foot_ori
        l_foot_quat = matrix_to_quaternion(L_foot_ori)
        l_ankle_quat = rot_l_ankle @ [l_foot_quat[0], l_foot_quat[1], l_foot_quat[2]]
        T_l_foot = quaternion_to_matrix([-l_ankle_quat[0], -l_ankle_quat[2], -l_ankle_quat[1], l_foot_quat[3]])
        T_l_foot_shank = T_l_shank.T @ T_l_foot
        l_ankle = matrix_to_quaternion(T_l_foot_shank)
        self.data.qpos[29] = l_ankle[3]
        self.data.qpos[30] = l_ankle[0]  # 外摆
        self.data.qpos[31] = l_ankle[1]
        self.data.qpos[32] = l_ankle[2]  # 弯曲
        '''角速度'''
        l_foot_angle_velocity = rot_l_ankle @ L_foot[1]
        qvel_l_ankle = l_foot_angle_velocity - l_shank_angle_velocity
        self.data.qvel[23] = -qvel_l_ankle[0]
        self.data.qvel[24] = -qvel_l_ankle[2]
        self.data.qvel[25] = -qvel_l_ankle[1]

        '''背部角度'''
        back_quaternion = Back[0]
        back_ori = quaternion_to_matrix(back_quaternion)
        Back_ori = self.back_ori_init.T @ back_ori
        back_quat = matrix_to_quaternion(Back_ori)
        BACK_quat = rot_back @ [back_quat[0], back_quat[1], back_quat[2]]
        T_back = quaternion_to_matrix([-BACK_quat[0], -BACK_quat[2], -BACK_quat[1], back_quat[3]])
        T_back_pelvis = T_pelvis.T @ T_back
        BACK = matrix_to_quaternion(T_back_pelvis)
        self.data.qpos[35] = BACK[3]
        self.data.qpos[36] = BACK[0]  # 外摆
        self.data.qpos[37] = BACK[1]
        self.data.qpos[38] = BACK[2]  # 弯曲
        '''角速度'''
        back_angle_velocity = rot_back @ Back[1]
        qvel_back= back_angle_velocity - pelvis_angle_velocity
        self.data.qvel[28] = -qvel_back[0]
        self.data.qvel[29] = -qvel_back[2]
        self.data.qvel[30] = -qvel_back[1]

        '''右肩关节角度'''
        r_arm_quaternion = R_arm[0]
        r_arm_ori = quaternion_to_matrix(r_arm_quaternion)
        R_arm_ori = self.r_arm_ori_init.T @ r_arm_ori
        r_arm_quat = matrix_to_quaternion(R_arm_ori)
        r_shoulder_quat = rot_r_shoulder @ [r_arm_quat[0], r_arm_quat[1], r_arm_quat[2]]
        T_r_arm = quaternion_to_matrix([-r_shoulder_quat[0], -r_shoulder_quat[2], -r_shoulder_quat[1], r_arm_quat[3]])
        T_r_arm_back = T_back.T @ T_r_arm
        r_shoulder = matrix_to_quaternion(T_r_arm_back)
        self.data.qpos[39] = r_shoulder[3]
        self.data.qpos[40] = r_shoulder[0]  # 外摆
        self.data.qpos[41] = r_shoulder[1]
        self.data.qpos[42] = r_shoulder[2]  # 弯曲
        '''角速度'''
        r_arm_angle_velocity = rot_r_shoulder @ R_arm[1]
        qvel_r_shoulder = r_arm_angle_velocity - back_angle_velocity
        self.data.qvel[31] = -qvel_r_shoulder[0]
        self.data.qvel[32] = -qvel_r_shoulder[2]
        self.data.qvel[33] = -qvel_r_shoulder[1]

        '''右肘关节角度'''
        r_forearm_quaternion = R_forearm[0]
        r_forearm_ori = quaternion_to_matrix(r_forearm_quaternion)
        R_forearm_ori = self.r_forearm_ori_init.T @ r_forearm_ori
        r_forearm_quat = matrix_to_quaternion(R_forearm_ori)
        r_elbow_quat = rot_r_forearm @ [r_forearm_quat[0], r_forearm_quat[1], r_forearm_quat[2]]
        T_r_forearm = quaternion_to_matrix([-r_elbow_quat[0], -r_elbow_quat[2], -r_elbow_quat[1], r_forearm_quat[3]])
        T_r_forearm_arm = T_r_arm.T @ T_r_forearm
        r_elbow = matrix_to_quaternion(T_r_forearm_arm)
        self.data.qpos[43] = abs(r_elbow[3])
        # self.data.qpos[44] = r_elbow[0]
        # self.data.qpos[45] = r_elbow[1]
        self.data.qpos[46] = abs(r_elbow[2])  # 弯曲
        '''角速度'''
        r_forearm_angle_velocity = rot_r_forearm @ R_forearm[1]
        qvel_r_elbow = r_forearm_angle_velocity[1] - r_arm_angle_velocity[1]
        self.data.qvel[36] = -qvel_r_elbow

        '''左肩关节角度'''
        l_arm_quaternion = L_arm[0]
        l_arm_ori = quaternion_to_matrix(l_arm_quaternion)
        L_arm_ori = self.l_arm_ori_init.T @ l_arm_ori
        l_arm_quat = matrix_to_quaternion(L_arm_ori)
        l_shoulder_quat = rot_l_shoulder @ [l_arm_quat[0], l_arm_quat[1], l_arm_quat[2]]
        T_l_arm = quaternion_to_matrix([-l_shoulder_quat[0], -l_shoulder_quat[2], -l_shoulder_quat[1], l_arm_quat[3]])
        T_l_arm_back = T_back.T @ T_l_arm
        l_shoulder = matrix_to_quaternion(T_l_arm_back)
        self.data.qpos[50] = l_shoulder[3]
        self.data.qpos[51] = l_shoulder[0]  # 外摆
        self.data.qpos[52] = l_shoulder[1]
        self.data.qpos[53] = l_shoulder[2]  # 弯曲
        '''角速度'''
        l_arm_angle_velocity = rot_l_shoulder @ L_arm[1]
        qvel_l_shoulder = l_arm_angle_velocity - back_angle_velocity
        self.data.qvel[40] = -qvel_l_shoulder[0]
        self.data.qvel[41] = -qvel_l_shoulder[2]
        self.data.qvel[42] = -qvel_l_shoulder[1]

        '''左肘关节角度'''
        l_forearm_quaternion = L_forearm[0]
        l_forearm_ori = quaternion_to_matrix(l_forearm_quaternion)
        L_forearm_ori = self.l_forearm_ori_init.T @ l_forearm_ori
        l_forearm_quat = matrix_to_quaternion(L_forearm_ori)
        l_elbow_quat = rot_l_forearm @ [l_forearm_quat[0], l_forearm_quat[1], l_forearm_quat[2]]
        T_l_forearm = quaternion_to_matrix([-l_elbow_quat[0], -l_elbow_quat[2], -l_elbow_quat[1], l_forearm_quat[3]])
        T_l_forearm_arm = T_l_arm.T @ T_l_forearm
        l_elbow = matrix_to_quaternion(T_l_forearm_arm)
        self.data.qpos[54] = abs(l_elbow[3])
        # self.data.qpos[55] = l_elbow[0]  # 外摆
        # self.data.qpos[56] = l_elbow[1]
        self.data.qpos[57] = abs(l_elbow[2])  # 弯曲
        '''角速度'''
        l_forearm_angle_velocity = rot_l_forearm @ L_forearm[1]
        qvel_l_elbow = l_forearm_angle_velocity[1] - l_arm_angle_velocity[1]
        self.data.qvel[45] = -qvel_l_elbow

        mujoco.mj_forward(self.model, self.data)


if __name__ == '__main__':
    print('process init')
    read_imu = imu_posture()

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:5555")

    while True:
        read_imu.read_imu_data_right()
        read_imu.read_imu_data_left()
        if read_imu.count < 300:
            read_imu.imu_calibration()  # 初始化
        elif read_imu.count == 300:
            print('calibration')
        # elif 300 < read_imu.count < 1000:
        #     read_imu.get_rawdata()  # 记录数据
        # elif read_imu.count == 1000:
        #     read_imu.calibration_position()
        #     print('start')
        #     # threading.Thread(target=read_imu.calibration_position).start()
        else:
            read_imu.imu_data_solving()  # 正常工作
            publisher.send_multipart(
                [read_imu.data.qpos.tobytes(), read_imu.data.qvel.tobytes(), read_imu.position_dict.tobytes(), read_imu.posture_dict.tobytes()])


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
    #                 publisher.send_multipart([read_imu.data.qpos.tobytes(), read_imu.position_dict.tobytes(), read_imu.posture_dict.tobytes()])





