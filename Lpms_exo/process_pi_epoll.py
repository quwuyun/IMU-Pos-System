"""
7个蓝牙IMU解算版本：主线程+解算线程

1. epoll事件实现：使用epoll替代线程轮询，每个串口对应一个fd，主线程等待fd就绪事件并处理读取，减少CPU占用和延迟。
2. 解算线程优化：解算线程只处理最新一帧数据，使用queue.Queue(maxsize=1)实现生产者-消费者模式，避免数据积压和过时解算。
3. 异常处理和自动重连：在读取串口数据时捕获异常（如串口断开），尝试自动重连并重新注册fd，提高系统鲁棒性。
4. 代码结构调整：将读取和解算逻辑分离，主线程专注于I/O和数据传递，解算线程专注于标定和解算，职责清晰，易于维护和扩展。
5. 性能监控：增加I/O和解算耗时统计，定期打印平均耗时，便于性能分析和优化。
"""

import serial
import time
import struct
import zmq
import csv
import numpy as np
import mujoco       # pip install mujoco==2.3.7    pip install mujoco-python-viewer
from scipy.spatial.transform import Rotation
import linuxfd,signal,select
import threading
import queue
import copy

R_thigh = [[0, 0, 0, 1], [0, 0, 0]]
R_shank = [[0, 0, 0, 1], [0, 0, 0]]
R_foot = [[0, 0, 0, 1], [0, 0, 0]]
L_thigh = [[0, 0, 0, 1], [0, 0, 0]]
L_shank = [[0, 0, 0, 1], [0, 0, 0]]
L_foot = [[0, 0, 0, 1], [0, 0, 0]]
Pelvis = [[0, 0, 0, 1], [0, 0, 0]]

ENABLE_VIEWER = True

port0 = '/dev/rfcomm0'
port1 = '/dev/rfcomm1'
port2 = '/dev/rfcomm2'
port3 = '/dev/rfcomm3'
port4 = '/dev/rfcomm4'
port5 = '/dev/rfcomm5'
port6 = '/dev/rfcomm6'

def quaternion_to_matrix(quaternion):
    r = Rotation.from_quat(quaternion)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def matrix_to_quaternion(rotation_matrix):
    r = Rotation.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
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
                ser = serial.Serial(port, baudrate, timeout=0.01)
                ser.reset_input_buffer()  # 清空缓存，避免旧数据堆积
                self.ser_list.append(ser)
                self.buf_list.append(b"")  # 每个串口对应一个缓存
                print(f"成功连接 {port}")
            except Exception as e:
                raise RuntimeError(f"连接{port}失败: {e}")

        self.imu_data_dict = {}
        for i in range(self.imu_num):
            self.imu_data_dict[i] = [[0, 0, 0, 1], [0, 0, 0]]

        # 每个 IMU 的最近接收时间（用于判断是否所有 IMU 已有数据）
        self.last_seen = [0.0 for _ in range(self.imu_num)]

        # 标定控制
        self.calibrating = False
        self.calibration_done = False
        self.calibration_count = 0
        self.calibration_cycles = 300
        # 判定 IMU 最近数据可用的时限（秒）
        self.seen_timeout = 0.1

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

        self.count = 0
        self.time = time.time()
        self.cycle_times = []

        self.model_file = './model/new/walk_new_quat_body0901.xml'
        self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
        self.data = mujoco.MjData(self.model)

        # 解算线程相关：队列只保留最新一帧（maxsize=1）
        self.solve_queue = queue.Queue(maxsize=1)
        self.solve_count = 0
        self.cycle_times_solve = []
        # 内部 publisher，解算线程负责发布
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind("tcp://*:5555")
        self.solver_thread = threading.Thread(target=self._solver_loop, daemon=True)
        self.solver_thread.start()

        # 使用 epoll 多路复用替代线程读取
        self.epoll = select.epoll()
        self.fd_map = {}  # fd -> imu_idx
        for i, ser in enumerate(self.ser_list):
            try:
                fd = ser.fileno()
                self.fd_map[fd] = i
                self.epoll.register(fd, select.EPOLLIN)
            except Exception:
                pass

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

        # 时间戳（uint16_t，1/100秒）
        timestamp_raw = int.from_bytes(payload_bytes[0:2], 'little')
        data["timestamp"] = timestamp_raw / 100.0

        # 解析float32小端数据
        floats = []
        for i in range(4, len(payload_bytes), 4):
            if i + 4 <= len(payload_bytes):
                val = struct.unpack('<f', payload_bytes[i:i + 4])[0]
                floats.append(val)
        try:
            data["gyro"] = tuple(floats[0:3])  # 角速度 rad/s
            data["quat"] = tuple(floats[3:7])  # 四元数(w,x,y,z)
        except IndexError as e:
            print(f"解析异常: {e} | payload长度: {len(payload_bytes)}")
        return data

    def _reopen(self, imu_idx, delay=1.0):
        port = self.ports[imu_idx]
        try:
            old = self.ser_list[imu_idx]
            # 尝试取消注册旧 fd（如果存在）以避免遗留映射
            try:
                old_fd = old.fileno()
                try:
                    self.epoll.unregister(old_fd)
                except Exception:
                    pass
                if old_fd in self.fd_map:
                    del self.fd_map[old_fd]
            except Exception:
                old_fd = None

            try:
                old.close()
            except Exception:
                pass
            # 重连期间认为该 IMU 暂不可用
            try:
                self.last_seen[imu_idx] = 0.0
            except Exception:
                pass

            time.sleep(delay)
            ser = serial.Serial(port, self.baudrate, timeout=0.01)
            ser.reset_input_buffer()
            self.ser_list[imu_idx] = ser
            self.buf_list[imu_idx] = b""

            # 注册新 fd 并更新映射
            try:
                new_fd = ser.fileno()
                self.fd_map[new_fd] = imu_idx
                self.epoll.register(new_fd, select.EPOLLIN)
            except Exception as e:
                print(f"[IMU{imu_idx}] 新 fd 注册失败: {e}")

            print(f"[IMU{imu_idx}] 重新连接成功 {port} (fd {new_fd})")
        except Exception as e:
            print(f"[IMU{imu_idx}] 重新连接失败 {port}: {e}")

    def _handle_fd_event(self, fd):
        """处理单个就绪 fd 的读取与解析"""
        imu_idx = self.fd_map.get(fd)
        if imu_idx is None:
            return
        ser = self.ser_list[imu_idx]
        try:
            n = ser.in_waiting
            if n > 0:
                chunk = ser.read(n)
                buf = self.buf_list[imu_idx] + chunk
                frames, buf = self.extract_frames(buf)
                self.buf_list[imu_idx] = buf

                if frames:
                    latest_frame = frames[-1]
                    data_len = int.from_bytes(latest_frame[5:7], 'little')
                    payload = latest_frame[7:7 + data_len]
                    parsed_data = self.parse_lpms_payload(payload)
                    if all(key in parsed_data for key in ["gyro", "quat"]):
                        q = parsed_data["quat"]
                        q = [q[1], q[2], q[3], q[0]]  # (x,y,z,w)
                        g = list(parsed_data["gyro"])  # 角速度
                        self.imu_data_dict[imu_idx] = [q, g]
                        # 更新最近接收时间
                        try:
                            self.last_seen[imu_idx] = time.time()
                        except Exception:
                            pass
        except (OSError, serial.SerialException) as e:
            print(f"[IMU{imu_idx}] 串口异常: {e} ({self.ports[imu_idx]})，尝试重连...")
            self._reopen(imu_idx)

    def read_imu_data(self):
        """使用 epoll.poll 获取所有就绪串口并处理，返回数据字典"""
        # 记录 I/O 开始时间，用于测量纯 I/O 耗时（ms）
        t0 = time.perf_counter()

        # 等待很短时间，降低默认延迟（从 10ms 降到 2ms）
        try:
            events = self.epoll.poll(0)
        except Exception:
            events = []
        for fd, event in events:
            if event & (select.EPOLLIN | select.EPOLLPRI):
                self._handle_fd_event(fd)
            elif event & (select.EPOLLHUP | select.EPOLLERR):
                # 出错或挂起，尝试重连对应 imu
                imu_idx = self.fd_map.get(fd)
                if imu_idx is not None:
                    print(f"[IMU{imu_idx}] fd错误，尝试重连")
                    self._reopen(imu_idx)
        t1 = time.perf_counter()
        io_elapsed_ms = (t1 - t0) * 1000.0
        # 保存最近的 I/O 耗时用于诊断
        if not hasattr(self, 'cycle_times_io'):
            self.cycle_times_io = []
        self.cycle_times_io.append(io_elapsed_ms)

        self.count += 1
        # 每100次打印一次 I/O 平均耗时（不包含解算）
        if self.count % 1000 == 0:
            last100 = self.cycle_times_io[-100:]
            avg_io = sum(last100) / len(last100) if last100 else 0.0
            # print(f"[epoll] Cycle {self.count}: IO last {len(last100)} avg {avg_io:.2f} ms | last {io_elapsed_ms:.2f} ms")

        # 将最新数据投递给解算线程（非阻塞替换旧数据）
        try:
            if self.solve_queue.full():
                try:
                    _ = self.solve_queue.get_nowait()
                except Exception:
                    pass
            # 发送一个浅拷贝，避免并发修改带来的问题
            self.solve_queue.put_nowait(copy.deepcopy(self.imu_data_dict))
        except Exception:
            pass

        return self.imu_data_dict

    def _solver_loop(self):
        """后台解算线程：从队列获取最新 IMU 数据，执行校准/解算并发布结果"""
        while True:
            try:
                data = self.solve_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            # 把队列数据拷贝到共享字典
            try:
                for k, v in data.items():
                    self.imu_data_dict[k] = v
            except Exception:
                pass

            # 判断所有 IMU 是否在最近一段时间内都有数据
            now = time.time()
            all_seen = True
            for i in range(self.imu_num):
                ts = self.last_seen[i] if i < len(self.last_seen) else 0.0
                if ts <= 0.0 or (now - ts) > self.seen_timeout:
                    all_seen = False
                    break

            did_work = False
            t0 = time.perf_counter()

            if not self.calibration_done:
                # 需要先进行标定
                if not all_seen and not self.calibrating:
                    # 等待所有 IMU 首次就绪后开始标定
                    pass
                else:
                    # 如果已经开始标定（或刚好全部就绪），继续/开始标定
                    if not self.calibrating:
                        print("[solver] 所有 IMU 就绪，开始标定")
                        self.calibrating = True
                        self.calibration_count = 0

                    # 执行一次标定步骤（标定一旦开始即使部分 IMU 丢失也不中止）
                    self.calibration_count += 1
                    self.solve_count += 1
                    self.imu_calibration()
                    did_work = True
                    time.sleep(0.0)

                    if self.calibration_count >= self.calibration_cycles:
                        self.calibration_done = True
                        self.calibrating = False
                        print('[solver] calibration完成')

            else:
                # 标定已完成；即便部分 IMU 丢失也继续解算（不重新标定）
                self.solve_count += 1
                self.imu_data_solving()
                try:
                    self.publisher.send_multipart([self.data.qpos.tobytes()])
                except Exception:
                    pass
                did_work = True

            # 统计并打印耗时（仅对实际执行的解算/标定步骤计时）
            if did_work:
                t1 = time.perf_counter()
                solve_elapsed_ms = (t1 - t0) * 1000.0
                self.cycle_times_solve.append(solve_elapsed_ms)
                if self.solve_count % 1000 == 0:
                    last100 = self.cycle_times_solve[-100:]
                    avg = sum(last100) / len(last100) if last100 else 0.0
                    print(f"[solver] Cycle {self.solve_count}: solve last {len(last100)} avg {avg:.2f} ms | last {solve_elapsed_ms:.2f} ms")

    def close(self):
        """关闭 epoll 和所有串口"""
        try:
            self.epoll.close()
        except Exception:
            pass
        for fd in list(self.fd_map.keys()):
            try:
                self.epoll.unregister(fd)
            except Exception:
                pass
        for ser in self.ser_list:
            try:
                if ser.is_open:
                    ser.close()
            except Exception:
                pass
        print("\n🔌 所有串口已关闭")


    def imu_calibration(self):
        imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        Pelvis, R_thigh, L_thigh, R_shank, L_shank, R_foot, L_foot, = imu_data_list
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

    def imu_data_solving(self):
        imu_data_list = [self.imu_data_dict[i] for i in range(0, self.imu_num)]
        Pelvis, R_thigh, L_thigh, R_shank, L_shank, R_foot, L_foot, = imu_data_list
        pelvis_quaternion = Pelvis[0]
        pelvis_ori = quaternion_to_matrix(pelvis_quaternion)
        Pelvis_ori = self.pelvis_ori_init.T @ pelvis_ori
        pelvis_quat = matrix_to_quaternion(Pelvis_ori)
        PELVIS_quat = self.rot_pelvis @ [pelvis_quat[0], pelvis_quat[1], pelvis_quat[2]]
        self.data.qpos[3] = pelvis_quat[3]
        self.data.qpos[4] = -PELVIS_quat[0]
        self.data.qpos[5] = -PELVIS_quat[2]
        self.data.qpos[6] = -PELVIS_quat[1]
        T_pelvis = quaternion_to_matrix([-PELVIS_quat[0], -PELVIS_quat[2], -PELVIS_quat[1], pelvis_quat[3]])
        pelvis_angle_velocity = self.rot_pelvis @ Pelvis[1]
        qvel_pelvis = pelvis_angle_velocity
        self.data.qvel[3] = -qvel_pelvis[0]
        self.data.qvel[4] = -qvel_pelvis[2]
        self.data.qvel[5] = -qvel_pelvis[1]

        r_thigh_quaternion = R_thigh[0]
        r_thigh_ori = quaternion_to_matrix(r_thigh_quaternion)
        R_thigh_ori = self.r_thigh_ori_init.T @ r_thigh_ori
        r_thigh_quat = matrix_to_quaternion(R_thigh_ori)
        r_hip_quat = self.rot_r_hip @ [r_thigh_quat[0], r_thigh_quat[1], r_thigh_quat[2]]
        T_r_thigh = quaternion_to_matrix([-r_hip_quat[0], -r_hip_quat[2], -r_hip_quat[1], r_thigh_quat[3]])
        T_r_thigh_pelvis = T_pelvis.T @ T_r_thigh
        r_hip = matrix_to_quaternion(T_r_thigh_pelvis)
        self.data.qpos[7] = r_hip[3]
        self.data.qpos[8] = r_hip[0]
        self.data.qpos[9] = r_hip[1]
        self.data.qpos[10] = r_hip[2]
        r_thigh_angle_velocity = self.rot_r_hip @ R_thigh[1]
        qvel_r_hip = r_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[6] = -qvel_r_hip[0]
        self.data.qvel[7] = -qvel_r_hip[2]
        self.data.qvel[8] = -qvel_r_hip[1]

        r_shank_quaternion = R_shank[0]
        r_shank_ori = quaternion_to_matrix(r_shank_quaternion)
        R_shank_ori = self.r_shank_ori_init.T @ r_shank_ori
        r_shank_quat = matrix_to_quaternion(R_shank_ori)
        r_knee_quat = self.rot_r_shank @ [r_shank_quat[0], r_shank_quat[1], r_shank_quat[2]]
        T_r_shank = quaternion_to_matrix([-r_knee_quat[0], -r_knee_quat[2], -r_knee_quat[1], r_shank_quat[3]])
        T_r_shank_thigh = T_r_thigh.T @ T_r_shank
        r_knee = matrix_to_quaternion(T_r_shank_thigh)
        self.data.qpos[11] = abs(r_knee[3])
        self.data.qpos[14] = -abs(r_knee[2])
        r_shank_angle_velocity = self.rot_r_shank @ R_shank[1]
        qvel_r_knee = r_shank_angle_velocity[1] - r_thigh_angle_velocity[1]
        self.data.qvel[11] = -qvel_r_knee

        r_foot_quaternion = R_foot[0]
        r_foot_ori = quaternion_to_matrix(r_foot_quaternion)
        R_foot_ori = self.r_foot_ori_init.T @ r_foot_ori
        r_foot_quat = matrix_to_quaternion(R_foot_ori)
        r_ankle_quat = self.rot_r_ankle @ [r_foot_quat[0], r_foot_quat[1], r_foot_quat[2]]
        T_r_foot = quaternion_to_matrix([-r_ankle_quat[0], -r_ankle_quat[2], -r_ankle_quat[1], r_foot_quat[3]])
        T_r_foot_shank = T_r_shank.T @ T_r_foot
        r_ankle = matrix_to_quaternion(T_r_foot_shank)
        self.data.qpos[15] = r_ankle[3]
        self.data.qpos[16] = r_ankle[0]
        self.data.qpos[17] = r_ankle[1]
        self.data.qpos[18] = r_ankle[2]
        r_foot_angle_velocity = self.rot_r_ankle @ R_foot[1]
        qvel_r_ankle = r_foot_angle_velocity - r_shank_angle_velocity
        self.data.qvel[12] = -qvel_r_ankle[0]
        self.data.qvel[13] = -qvel_r_ankle[2]
        self.data.qvel[14] = -qvel_r_ankle[1]

        l_thigh_quaternion = L_thigh[0]
        l_thigh_ori = quaternion_to_matrix(l_thigh_quaternion)
        L_thigh_ori = self.l_thigh_ori_init.T @ l_thigh_ori
        l_thigh_quat = matrix_to_quaternion(L_thigh_ori)
        l_hip_quat = self.rot_l_hip @ [l_thigh_quat[0], l_thigh_quat[1], l_thigh_quat[2]]
        T_l_thigh = quaternion_to_matrix([-l_hip_quat[0], -l_hip_quat[2], -l_hip_quat[1], l_thigh_quat[3]])
        T_l_thigh_pelvis = T_pelvis.T @ T_l_thigh
        l_hip = matrix_to_quaternion(T_l_thigh_pelvis)
        self.data.qpos[21] = l_hip[3]
        self.data.qpos[22] = l_hip[0]
        self.data.qpos[23] = l_hip[1]
        self.data.qpos[24] = l_hip[2]
        l_thigh_angle_velocity = self.rot_l_hip @ L_thigh[1]
        qvel_l_hip = l_thigh_angle_velocity - pelvis_angle_velocity
        self.data.qvel[17] = -qvel_l_hip[0]
        self.data.qvel[18] = -qvel_l_hip[2]
        self.data.qvel[19] = -qvel_l_hip[1]

        l_shank_quaternion = L_shank[0]
        l_shank_ori = quaternion_to_matrix(l_shank_quaternion)
        L_shank_ori = self.l_shank_ori_init.T @ l_shank_ori
        l_shank_quat = matrix_to_quaternion(L_shank_ori)
        l_knee_quat = self.rot_l_shank @ [l_shank_quat[0], l_shank_quat[1], l_shank_quat[2]]
        T_l_shank = quaternion_to_matrix([-l_knee_quat[0], -l_knee_quat[2], -l_knee_quat[1], l_shank_quat[3]])
        T_l_shank_thigh = T_l_thigh.T @ T_l_shank
        l_knee = matrix_to_quaternion(T_l_shank_thigh)
        self.data.qpos[25] = abs(l_knee[3])
        self.data.qpos[28] = -abs(l_knee[2])
        l_shank_angle_velocity = self.rot_l_shank @ L_shank[1]
        qvel_l_knee = l_shank_angle_velocity[1] - l_thigh_angle_velocity[1]
        self.data.qvel[22] = -qvel_l_knee

        l_foot_quaternion = L_foot[0]
        l_foot_ori = quaternion_to_matrix(l_foot_quaternion)
        L_foot_ori = self.l_foot_ori_init.T @ l_foot_ori
        l_foot_quat = matrix_to_quaternion(L_foot_ori)
        l_ankle_quat = self.rot_l_ankle @ [l_foot_quat[0], l_foot_quat[1], l_foot_quat[2]]
        T_l_foot = quaternion_to_matrix([-l_ankle_quat[0], -l_ankle_quat[2], -l_ankle_quat[1], l_foot_quat[3]])
        T_l_foot_shank = T_l_shank.T @ T_l_foot
        l_ankle = matrix_to_quaternion(T_l_foot_shank)
        self.data.qpos[29] = l_ankle[3]
        self.data.qpos[30] = l_ankle[0]
        self.data.qpos[31] = l_ankle[1]
        self.data.qpos[32] = l_ankle[2]
        l_foot_angle_velocity = self.rot_l_ankle @ L_foot[1]
        qvel_l_ankle = l_foot_angle_velocity - l_shank_angle_velocity
        self.data.qvel[23] = -qvel_l_ankle[0]
        self.data.qvel[24] = -qvel_l_ankle[2]
        self.data.qvel[25] = -qvel_l_ankle[1]


if __name__ == '__main__':
    print('process parallel init')
    IMU_PORTS = [port0, port1, port2, port3, port4, port5, port6]
    read_imu = imu_posture(ports=IMU_PORTS)
    while True:
        t0 = time.perf_counter()
        read_imu.read_imu_data()
        # 主循环只负责 I/O 并将数据发送给解算线程
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        read_imu.cycle_times.append(elapsed_ms)
        if read_imu.count % 1000 == 0:
            last100 = read_imu.cycle_times[-100:]
            avg = sum(last100) / len(last100)
            # print(f"Cycle {read_imu.count}: IO-only last {len(last100)} avg {avg:.2f} ms | last {elapsed_ms:.2f} ms")
