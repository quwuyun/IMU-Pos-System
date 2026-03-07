import linuxfd, signal, select
import serial
import time
import struct
from scipy.spatial.transform import Rotation

ser = serial.Serial('/dev/ttyAMA0', 921600, timeout=0.01)  # 115200,230400,460800
txData = b'\xFF\xFF\x80\n'
ser.write(txData)
ti = time.time()
count = 0

imu_num = 6  # imu数量
imu_id = 0x80 + imu_num
imu_data_dict = {}


def quaternion_to_matrix(quaternion):
    # 创建四元数对象(x, y, z, w)
    r = Rotation.from_quat(quaternion)
    # 将四元数转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    return rotation_matrix


# while True:
#     readlen = 38 * imu_num  # 前16位四元数，17—22角速度，23-28线加速度a_raw，29-34线加速度a_real，35-38校验位：0xff,0xff,id,/n
#     data = ser.read(readlen)
#     l = len(data)
#     if l == readlen:
#         id = data[l - 2]
#         if id == imu_id:
#             for i in range(1, imu_num + 1):
#                 start_index = (i - 1) * 34 + (i - 1) * 4
#                 end_index = start_index + 34
#                 result = struct.unpack('4f 3h 3h 3h', data[start_index: end_index])  # 解码
#                 q_raw = list(result[0:4])
#                 q = [q_raw[1], q_raw[2], q_raw[3], q_raw[0]]
#                 g_raw = list(result[4:7])
#                 g = [x / 938.734 for x in g_raw]
#                 a_raw = list(result[7:10])
#                 a_raw = ([a_raw[0] / 208.980, a_raw[1] / 208.980, a_raw[2] / 208.980])
#                 # a_real = list(result[10:13])
#                 # a_real = ([a_real[0] / 208.980, a_real[1] / 208.980, a_real[2] / 208.980])
#                 Matrix = quaternion_to_matrix(q)
#                 a_real = (a_raw - Matrix.T @ [0.0, 0.0, 9.8]).tolist()
#                 imu_data = [q, g, a_raw, a_real]
#                 imu_data_dict[i - 1] = imu_data  # 第k个imu的数据放在imu_data_dict的第k位
#                 # print(imu_data)
#             # print(imu_data_dict[0])
#             count = count + 1
#         else:
#             print('wrong id =', id)
#             print('data is :')
#             for i in range(0, l):
#                 print('%#x ' % data[i], end='')
#             print('')
#
#         if count % 100 == 0:
#             print('count =', count)
#             print('time =', time.time() - ti)
#             ti = time.time()
#
#     else:
#         print('err: received length=', l)
#
#     ser.write(txData)


# create special file objects
efd = linuxfd.eventfd(initval=0, nonBlocking=True)
sfd = linuxfd.signalfd(signalset={signal.SIGINT}, nonBlocking=True)
tfd = linuxfd.timerfd(rtc=True, nonBlocking=True)

# program timer and mask SIGINT
tfd.settime(1, 0.01)
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
    # t = time.time()
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
            readlen = 38 * imu_num  # 前16位四元数，17—22角速度，23-28线加速度a_raw，29-34线加速度a_real，35-38校验位：0xff,0xff,id,/n
            # print("{0:.5f}: READ_DATA_BEGIN".format(time.time()))
            data = ser.read(readlen)
            # print("{0:.5f}: READ_DATA_DONE".format(time.time()))
            l = len(data)
            if l == readlen:
                id = data[l - 2]
                if id == imu_id:
                    for i in range(1, imu_num + 1):
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
                        imu_data_dict[i - 1] = imu_data  # 第k个imu的数据放在imu_data_dict的第k位
                        # print(imu_data)
                    # print(imu_data_dict[0][0])
                    # print('')
                    # print("{0:.5f}: SOLVING_DATA_DONE".format(time.time()))
                    # print('')
                    count = count + 1
                else:
                    print('wrong id =', id)
                    print('data is :')
                    for i in range(0, l):
                        print('%#x ' % data[i], end='')
                    print('')

                if count % 100 == 0:
                    print('count =', count)
                    print('time =', time.time() - ti)
                    ti = time.time()

            else:
                print('err: received length=', l)
                # print('wrong data is :')
                # for i in range(0, l):
                #     print('0x{:02X} '.format(data[i]), end='')
                #     # print('%0x ' % data[i], end='')
                # print('')
                # time.sleep(0.05)
                # break
            # time.sleep(0.001)
            # print(count)
            ser.write(txData)

print("{0:.3f}: Goodbye!".format(time.time()))
