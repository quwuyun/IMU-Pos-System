import numpy as np
import math
import csv
import time
import threading


class get_axis_position:

    def __init__(self, file):
        self.DATASET_NUM = 650   # 采集数据数量
        self.ITER_STEP = 1e-5    # 迭代步长
        self.ITER_CNT = 100      # 迭代次数
        # self.imu_raw_data_1 = np.zeros((self.DATASET_NUM, 9))
        # self.imu_raw_data_2 = np.zeros((self.DATASET_NUM, 9))
        self.imu_raw_data = np.zeros((10, self.DATASET_NUM, 9))
        self.o1 = [0, 0, 0]
        self.o2 = [0, 0, 0]
        self.o3 = [0, 0, 0]
        self.o4 = [0, 0, 0]
        self.o5 = [0, 0, 0]
        self.o6 = [0, 0, 0]
        self.o7 = [0, 0, 0]
        self.o8 = [0, 0, 0]
        self.o9 = [0, 0, 0]
        self.o10 = [0, 0, 0]

        self.j1 = [0, 0, 0]
        self.j2 = [0, 0, 0]
        self.j3 = [0, 0, 0]
        self.file = 'output/' + str(file) + '.csv'

    def get_raw_data(self):
        """
        获取两个imu的数据, 这里只需要角速度
        raw_data = [acc_x, acc_y, acc_z, vel_x, vel_y, vel_z, vel_dot_x, vel_dot_y, vel_dot_z] 线加速度，角速度，角加速度
        """
        with open(self.file, 'rt') as file:
            reader = list(csv.reader(file, dialect='excel', delimiter=','))
            data = np.array(reader)
            for i in range(self.DATASET_NUM):
                self.imu_raw_data[0][i] = data[i][0:9]
                self.imu_raw_data[1][i] = data[i][10:19]
                self.imu_raw_data[2][i] = data[i][20:29]
                self.imu_raw_data[3][i] = data[i][30:39]
                self.imu_raw_data[4][i] = data[i][40:49]
                self.imu_raw_data[5][i] = data[i][50:59]
                self.imu_raw_data[6][i] = data[i][60:69]
                self.imu_raw_data[7][i] = data[i][70:79]
                self.imu_raw_data[8][i] = data[i][80:89]
                self.imu_raw_data[9][i] = data[i][90:99]

    def get_pos(self, input, params, output):
        """
        获取关节相对于两个imu的位置
        位置input = [imu_raw_data_1, imu_raw_data_2] 共18列
        """
        # 定义6个待求参数
        o1x = params[0]
        o1y = params[1]
        o1z = params[2]
        o2x = params[3]
        o2y = params[4]
        o2z = params[5]
        i = 0
        while i < input.shape[0]:
            # 角加速度计算值
            acc_joint1_x = input[i, 4] * (input[i, 3] * o1y - input[i, 4] * o1x) - input[i, 5] * (input[i, 5] * o1x - input[i, 3] * o1z) + (input[i, 7] * o1z - input[i, 8] * o1y)
            acc_joint1_y = input[i, 5] * (input[i, 4] * o1z - input[i, 5] * o1y) - input[i, 3] * (input[i, 3] * o1y - input[i, 4] * o1x) + (input[i, 8] * o1x - input[i, 6] * o1z)
            acc_joint1_z = input[i, 3] * (input[i, 5] * o1x - input[i, 3] * o1z) - input[i, 4] * (input[i, 4] * o1z - input[i, 5] * o1y) + (input[i, 6] * o1y - input[i, 7] * o1x)

            acc_joint2_x = input[i, 13] * (input[i, 12] * o2y - input[i, 13] * o2x) - input[i, 14] * (input[i, 14] * o2x - input[i, 12] * o2z) + (input[i, 16] * o2z - input[i, 17] * o2y)
            acc_joint2_y = input[i, 14] * (input[i, 13] * o2z - input[i, 14] * o2y) - input[i, 12] * (input[i, 12] * o2y - input[i, 13] * o2x) + (input[i, 17] * o2x - input[i, 15] * o2z)
            acc_joint2_z = input[i, 12] * (input[i, 14] * o2x - input[i, 12] * o2z) - input[i, 13] * (input[i, 13] * o2z - input[i, 14] * o2y) + (input[i, 15] * o2y - input[i, 16] * o2x)
            # 目标函数
            output[i, 0] = np.sqrt(pow((input[i, 0] - acc_joint1_x), 2) + pow((input[i, 1] - acc_joint1_y), 2) + pow((input[i, 2] - acc_joint1_z), 2)) - \
                           np.sqrt(pow((input[i, 9] - acc_joint2_x), 2) + pow((input[i, 10] - acc_joint2_y), 2) + pow((input[i, 11] - acc_joint2_z), 2))
            i += 1

    def get_axis(self, input, params, output):
        """
            获取关节相对于两个imu的方向
            轴input = [vel_x1, vel_y1, vel_z1, vel_x2, vel_y2, vel_z2] 共6列
        """
        theta_1 = params[0]
        theta_2 = params[1]
        phi_1 = params[2]
        phi_2 = params[3]
        i = 0
        while i < input.shape[0]:
            # 目标模型
            output[i, 0] = np.sqrt(pow((input[i, 1] * math.sin(phi_1) - input[i, 2] * math.cos(phi_1) * math.sin(theta_1)), 2) +
                                   pow((input[i, 2] * math.cos(phi_1) * math.cos(theta_1) - input[i, 0] * math.sin(phi_1)), 2) +
                                   pow((input[i, 0] * math.cos(phi_1) * math.sin(theta_1) - input[i, 1] * math.cos(phi_1) * math.cos(theta_1)), 2)) - \
                           np.sqrt(pow((input[i, 4] * math.sin(phi_2) - input[i, 5] * math.cos(phi_2) * math.sin(theta_2)), 2) +
                                   pow((input[i, 5] * math.cos(phi_2) * math.cos(theta_2) - input[i, 3] * math.sin(phi_2)), 2) +
                                   pow((input[i, 3] * math.cos(phi_2) * math.sin(theta_2) - input[i, 4] * math.cos(phi_2) * math.cos(theta_2)), 2))
            i += 1

    def get_jacobian(self, func, input, params, output):
        """
        获取高斯牛顿法迭代式子里的Jacobian
        """
        m = input.shape[0]  # 数据数量
        n = params.shape[0]  # 未知参数数量
        out0 = np.zeros((m, 1))
        out1 = np.zeros((m, 1))
        # param0 = np.zeros((n, 1))
        # param1 = np.zeros((n, 1))
        # output = np.zeros((m, n))
        k = 0
        while k < n:
            param0 = np.zeros(n)
            param1 = np.zeros(n)
            for j in range(n):
                param0[j] = params[j]
                param1[j] = params[j]
            param0[k] -= self.ITER_STEP
            param1[k] += self.ITER_STEP
            func(input, param0, out0)
            func(input, param1, out1)
            for i in range(m):
                output[i][k] = ((out1 - out0) / (2 * self.ITER_STEP))[i]
            # output.block(0, j, m, 1) = (out1 - out0) / (2 * ITER_STEP)
            k += 1

    def gauss_newton(self, func, input, output, params):
        """
        高斯牛顿法
        """
        m = input.shape[0]
        n = params.shape[0]
        jmat = np.zeros((m, n))
        r = np.zeros((m, 1))
        tmp = np.zeros((m, 1))
        pre_mse = 0.0
        i = 0
        while i < self.ITER_CNT:
            mse = 0.0
            func(input, params, tmp)
            r = output - tmp
            self.get_jacobian(func, input, params, jmat)

            # 均方误差
            mse = r.T @ r
            mse /= m
            if abs(mse - pre_mse) < 1e-8:
                break
            else:
                pre_mse = mse
                # 参数更新
                delta = np.linalg.inv(jmat.T @ jmat) @ jmat.T @ r
                print('i = ', i, ' mse = ', mse)
                params += delta.reshape(n)
                i += 1
        print("params:", params.T)

    def imu_joint_pos_data_fit(self):
        """
        计算关节位置的数据输入接口
        """
        # output = np.zeros((self.DATASET_NUM, 1))
        # params_pos = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        # # input = np.hstack((self.imu_raw_data_1, self.imu_raw_data_2))

        # r_knee
        input1 = np.hstack((self.imu_raw_data[2], self.imu_raw_data[1]))
        output1 = np.zeros((self.DATASET_NUM, 1))
        params_pos1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.gauss_newton(self.get_pos, input1, output1, params_pos1)
        self.o1 = [params_pos1[0], params_pos1[1], params_pos1[2]]      # r_thigh
        self.o2 = [params_pos1[3], params_pos1[4], params_pos1[5]]      # r_shank
        # l_knee
        input2 = np.hstack((self.imu_raw_data[4], self.imu_raw_data[3]))
        output2 = np.zeros((self.DATASET_NUM, 1))
        params_pos2 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.gauss_newton(self.get_pos, input2, output2, params_pos2)
        self.o3 = [params_pos2[0], params_pos2[1], params_pos2[2]]      # l_thigh
        self.o4 = [params_pos2[3], params_pos2[4], params_pos2[5]]      # l_shank
        # r_elbow
        input3 = np.hstack((self.imu_raw_data[7], self.imu_raw_data[6]))
        output3 = np.zeros((self.DATASET_NUM, 1))
        params_pos3 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.gauss_newton(self.get_pos, input3, output3, params_pos3)
        self.o5 = [params_pos3[0], params_pos3[1], params_pos3[2]]      # r_arm
        self.o6 = [params_pos3[3], params_pos3[4], params_pos3[5]]      # r_forearm
        # l_elbow
        input4 = np.hstack((self.imu_raw_data[9], self.imu_raw_data[8]))
        output4 = np.zeros((self.DATASET_NUM, 1))
        params_pos4 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.gauss_newton(self.get_pos, input4, output4, params_pos4)
        self.o7 = [params_pos4[0], params_pos4[1], params_pos4[2]]      # l_arm
        self.o8 = [params_pos4[3], params_pos4[4], params_pos4[5]]      # l_forearm
        # pelvis
        input5 = np.hstack((self.imu_raw_data[5], self.imu_raw_data[0]))
        output5 = np.zeros((self.DATASET_NUM, 1))
        params_pos5 = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.gauss_newton(self.get_pos, input5, output5, params_pos5)
        self.o9 = [params_pos5[0], params_pos5[1], params_pos5[2]]      # pelvis
        self.o10 = [params_pos5[3], params_pos5[4], params_pos5[5]]     # back

    def imu_joint_axis_data_fit(self):
        """
        计算关节轴向的数据输入接口
        """
        output = np.zeros((self.DATASET_NUM, 1))
        params_axis1 = np.array([0.5, 0.5, 0.5, 0.5])
        params_axis2 = np.array([0.5, 0.5, 0.5, 0.5])

        # input = np.hstack((self.imu_raw_data_1[:, 3:6], self.imu_raw_data_2[:, 3:6]))
        input1 = np.hstack((self.imu_raw_data[0][:, 3:6], self.imu_raw_data[1][:, 3:6]))
        self.gauss_newton(self.get_axis, input1, output, params_axis1)
        self.j1 = [math.cos(params_axis1[2]) * math.cos(params_axis1[0]), math.cos(params_axis1[2]) * math.sin(params_axis1[0]), math.sin(params_axis1[2])]
        self.j2 = [math.cos(params_axis1[3]) * math.cos(params_axis1[1]), math.cos(params_axis1[3]) * math.sin(params_axis1[1]), math.sin(params_axis1[3])]
        # print('j1 = ', j1)
        # print('j2 = ', j2)
        input2 = np.hstack((self.imu_raw_data[2][:, 3:6], self.imu_raw_data[1][:, 3:6]))
        self.gauss_newton(self.get_axis, input2, output, params_axis2)
        self.j2 = [math.cos(params_axis2[2]) * math.cos(params_axis2[0]), math.cos(params_axis2[2]) * math.sin(params_axis2[0]), math.sin(params_axis2[2])]
        self.j3 = [math.cos(params_axis2[3]) * math.cos(params_axis2[1]), math.cos(params_axis2[3]) * math.sin(params_axis2[1]), math.sin(params_axis2[3])]
        # print('j2 = ', j2)
        # print('j3 = ', j3)

    def calculate_axis(self):
        self.get_raw_data()
        self.imu_joint_axis_data_fit()
        print('j1 = ', self.j1)
        print('j2 = ', self.j2)
        print('j3 = ', self.j3)
        return self.j1, self.j2, self.j3

    def calculate_position(self):
        self.get_raw_data()
        self.imu_joint_pos_data_fit()
        print('r_thigh = ', self.o1)
        print('r_shank = ', self.o2)
        print('l_thigh = ', self.o3)
        print('l_shank = ', self.o4)
        print('r_arm = ', self.o5)
        print('r_forearm = ', self.o6)
        print('l_arm = ', self.o7)
        print('l_forearm = ', self.o8)
        print('pelvis = ', self.o9)
        print('back = ', self.o10)
        # np.savetxt(r'output/position1' + '.txt', self.o1, fmt='%f', delimiter=',')
        # np.savetxt(r'output/position2' + '.txt', self.o2, fmt='%f', delimiter=',')
        # np.savetxt(r'output/position3' + '.txt', self.o3, fmt='%f', delimiter=',')
        return self.o1, self.o2, self.o3, self.o4, self.o5, self.o6, self.o7, self.o8, self.o9, self.o10


if __name__ == '__main__':
    # time.sleep(5)
    axis_position = get_axis_position('data')
    axis_position.calculate_position()



