"""
部署测试
模拟测试37维度输入观测，5个IMU
"""

import linuxfd,signal,select
import numpy as np
import time
from datetime import datetime
import logging

from process_pi import *
from sensor_to_obs_bluetooth import *

import torch


INPUT_DIM = 65
OUTPUT_DIM = 30
IMU_PORTS = [port0, port1, port2, port3, port4]

class Process:
    def __init__(self):
        # IMU
        self.read_imu = imu_posture(ports=IMU_PORTS)

        # 加载模型
        self.use_gpu = False  # GPU
        self.device = torch.device("cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu")
        self.total_input = np.zeros(INPUT_DIM)
        self.total_input_tensor = torch.tensor(self.total_input, dtype=torch.float32, device=self.device)
        self.action = np.zeros(OUTPUT_DIM)

        # 初始化电机


    def get_imu(self):
        self.read_imu.read_imu_data()
        if self.read_imu.count < 300:
            self.read_imu.imu_calibration()  # 初始化
        elif self.read_imu.count == 300:
            print('calibration完成')
            # print('\nStart!!!\n')
        else:
            self.read_imu.imu_data_solving()  # 正常工作

    def get_obs(self):
        """传感器输入获取实际观测"""
        joint_pos, joint_vel, root_pos_w, root_vel_w = imu_to_obs(self.read_imu.data.qpos, self.read_imu.data.qvel)  # 65维度
        self.total_input = np.concatenate([joint_pos, joint_vel, root_pos_w, root_vel_w])
        self.total_input_tensor = torch.tensor(self.total_input, dtype=torch.float32, device=self.device)
        print(f"右髋y轴：{180/np.pi*self.total_input[1]}度\t右膝y轴：{180/np.pi*self.total_input[6]}度")
        # print(f"obs输入维度:", self.total_input.shape)

        return self.total_input_tensor

    # def inference(self):
    #     """推理"""
    #
    #     with torch.no_grad():
    #         # 检测归一化
    #         if self.obs_normalizer is not None:
    #             obs_normalized = self.obs_normalizer.normalize(self.total_input_tensor)
    #         else:
    #             obs_normalized = self.total_input_tensor
    #         obs_normalized = obs_normalized.float()
    #
    #         with torch.inference_mode():
    #             action = self.actor_model(obs_normalized)
    #
    #     # 去掉 batch 维度
    #     if action.dim() == 2:
    #         action = action.squeeze(0)
    #
    #     self.knee_torque_r = action[28].item()
    #     self.knee_torque_l = action[29].item()
    #     print(f"膝关节力矩：{[self.knee_torque_r, self.knee_torque_l]}")
    #     self.step += 1
    #
    #     return [self.knee_torque_r, self.knee_torque_l]


    # def low_pass_filter(self, left_torque, right_torque, alpha=0.2):
    #     """
    #     双通道低通滤波
    #     Args:
    #         left_torque: 左关节力矩
    #         right_torque: 右关节力矩
    #         alpha: 滤波系数 (0-1)，越小滤波越强
    #     Returns:
    #         (滤波后的左力矩, 滤波后的右力矩)
    #     """
    #     if self.first_run:
    #         self.last_values["left"] = left_torque
    #         self.last_values["right"] = right_torque
    #         self.first_run = False
    #     else:
    #         self.last_values["left"] = alpha * left_torque + (1 - alpha) * self.last_values["left"]
    #         self.last_values["right"] = alpha * right_torque + (1 - alpha) * self.last_values["right"]
    #
    #     return self.last_values["left"], self.last_values["right"]
    #
    # def limit_torque(self, torque):
    #     if torque > 0.05:
    #         torque = 0.05
    #     elif torque < -0.05:
    #         torque = -0.05
    #     return torque
    #
    # def set_effort_motor(self, torques):
    #     try:
    #         # 官方SDK的SerialPort不提供列出端口的功能，可以手动打印提示
    #         # print("请确保 `MOTOR_CHANNELS` 配置中的串口设备路径正确 (例如 /dev/ttyUSB0)。")
    #         filtered_torque_r, filtered_torque_l = self.low_pass_filter(torques[0], torques[1])  # [-1,1]
    #         limit_torque_r = self.limit_torque(filtered_torque_r/5)  # [-0.04,0.04]
    #         limit_torque_l = self.limit_torque(filtered_torque_l/5)
    #
    #         final_torque_r = limit_torque_r
    #         final_torque_l = 0
    #         print("电机发送外骨骼力矩:", final_torque_r, final_torque_l)
    #         self.saver_torque.save_row({"t_right_skrl": torques[0], "t_left_skrl": torques[1],
    #                              "t_right_filter": filtered_torque_r, "t_left_filter": filtered_torque_l,
    #                              "t_right_limit": limit_torque_r, "t_left_limit": limit_torque_l,
    #                              "t_right_final": final_torque_r, "t_left_final": final_torque_l,
    #                              })
    #
    #         # self.motor.send(0, 0, 0, 0, 0, final_torque_r)
    #         # self.motor.send(1, 0, 0, 0, 0, final_torque_l)
    #         # self.motor.check_status(0)
    #         # self.motor.check_status(1)
    #
    #     except (KeyboardInterrupt, InterruptedError):
    #         print("\n检测到用户中断操作 (Ctrl+C)。")
    #         logging.warning("用户中断操作。")
    #         # self.motor.emergency_stop()


    def main(self):
        print('process init')

        time.sleep(1)
        print("Process Start!\n")
        try:
            # create special file objects
            efd = linuxfd.eventfd(initval=0, nonBlocking=True)
            sfd = linuxfd.signalfd(signalset={signal.SIGINT}, nonBlocking=True)
            tfd = linuxfd.timerfd(rtc=True, nonBlocking=True)
            # program timer and mask SIGINT
            tfd.settime(1, 0.06)  # 第一次定时器间隔和以后所有定时器间隔
            signal.pthread_sigmask(signal.SIG_SETMASK, {signal.SIGINT})
            # create epoll instance and register special files
            epl = select.epoll()
            epl.register(efd.fileno(), select.EPOLLIN)
            epl.register(sfd.fileno(), select.EPOLLIN)
            epl.register(tfd.fileno(), select.EPOLLIN)
            # start main loop
            isrunning = True
            print("{0:.3f}: Hello!".format(time.time()))
            while isrunning:
                # block until epoll detects changes in the registered files
                events = epl.poll(-1)
                t = time.time()
                # iterate over occurred events
                for fd, event in events:
                    if fd == efd.fileno() and event & select.EPOLLIN:
                        # event file descriptor readable: read and exit loop
                        print("{0:.3f}: event file received update, exiting...".format(t))
                        efd.read()
                        isrunning = False
                    elif fd == sfd.fileno() and event & select.EPOLLIN:
                        # signal file descriptor readable: write to event file
                        siginfo = sfd.read()
                        if siginfo["signo"] == signal.SIGINT:
                            print("{0:.3f}: SIGINT received, notifying event file".format(t))
                            efd.write(1)
                    elif fd == tfd.fileno() and event & select.EPOLLIN:
                        # timer file descriptor readable: display that timer has expired
                        # print("{0:.3f}: timer has expired".format(t))
                        tfd.read()

                        time1 = time.time()

                        self.get_imu()
                        if self.read_imu.count > 300:
                            self.get_obs()

                        else:
                            print(f"初始标定中...{self.read_imu.count}/300")

                        time2 = time.time()
                        print(f"1个完整周期消耗时间: {time2-time1}")

                        # time.sleep(0.1)
        except KeyboardInterrupt:
            print("终止")


if __name__ == "__main__":
    process = Process()
    process.main()