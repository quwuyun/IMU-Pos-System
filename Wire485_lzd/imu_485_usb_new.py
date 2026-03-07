import serial
import time
import struct
from scipy.spatial.transform import Rotation

def quaternion_to_matrix(quaternion):
    # 创建四元数对象(x, y, z, w)
    r = Rotation.from_quat(quaternion)
    # 将四元数转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    return rotation_matrix


class imu485:

    def __init__(self, port, imu_num):
        self.count = 0
        self.imu_num = imu_num  # imu数量
        self.readlen = 0
        self.imu_id = 0x80 + self.imu_num
        self.imu_data_dict = {}  # 储存所有imu数据的字典
        for i in range(1, self.imu_num + 1):
            self.imu_data_dict[i - 1] = [[0, 0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.time = time.time()
        # port = '/dev/ttySC0'  # 串口号
        self.port = port
        self.ser = serial.Serial(self.port, 921600, timeout=0.01)
        self.txData = b'\xFF\xFF\x80\n'
        self.ser.write(self.txData)


    def read_imu_data(self):  # 采集imu数据
        self.readlen = 38 * self.imu_num  # 前16位四元数，17—22角速度，23-28线加速度a_raw，29-34线加速度a_real，35-38校验位：0xff,0xff,id,/n
        data = self.ser.read(self.readlen)
        l = len(data)
        if l == self.readlen:
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
                    self.imu_data_dict[i - 1] = imu_data  # 第k个imu的数据放在imu_data_dict的第k位
                    # print(imu_data)
                # print(imu_data_dict[1])
                self.count = self.count + 1
            else:
                print('wrong id =', id)
                print('data is :')
                for i in range(0, l):
                    print('%#x ' % data[i], end='')
                print('')

            if self.count % 100 == 0:
                print('count =', self.count)
                print('time =', time.time() - self.time)
                self.time = time.time()

        else:
            print('err: received length=', l)

        self.ser.write(self.txData)

        return self.imu_data_dict


if __name__ == "__main__":

    data_imu1 = imu485('/dev/ttySC0', 6)
    data_imu2 = imu485('/dev/ttySC1', 6)
    while True:
        data1 = data_imu1.read_imu_data()
        data2 = data_imu2.read_imu_data()
        # print('1', data1)
        # print('2', data2)
