import threading
import zmq
from mujoco_viewer.mujoco_viewer import *       # pip install mujoco-python-viewer
import csv
import datetime

ENABLE_VIEWER = True


class walking_sub:
    def __init__(self):
        model_file = './model/new/walk_new_quat_body0901.xml'
        self.model = mujoco.MjModel.from_xml_path(model_file)
        self.data = mujoco.MjData(self.model)  # for step sim

        mujoco.mj_resetData(self.model, self.data)
        viewer = MujocoViewer(self.model, self.data, width=1280, height=1024)
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        viewer.cam.distance = 3.0
        viewer.cam.lookat = [0, 0, 1]
        self.viewer = viewer

        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        # subscriber.connect("tcp://192.168.43.107:5555")
        subscriber.connect("tcp://192.168.10.70:5555")
        subscriber.setsockopt(zmq.SUBSCRIBE, b'')
        self.sub = subscriber

        self.position_dict = np.tile([0, 0, -0.1], 10)
        self.posture_dict = np.tile([1, 0, 0, 0, 1, 0, 0, 0, 1], 10)
        threading.Thread(target=self.recvProc).start()

        self.point_pelvis_imu = [0, 0, 0]
        self.point_r_thigh_imu = [0, 0, 0]
        self.point_r_shank_imu = [0, 0, 0]
        self.point_l_thigh_imu = [0, 0, 0]
        self.point_l_shank_imu = [0, 0, 0]
        self.point_back_imu = [0, 0, 0]
        self.point_r_arm_imu = [0, 0, 0]
        self.point_r_forearm_imu = [0, 0, 0]
        self.point_l_arm_imu = [0, 0, 0]
        self.point_l_forearm_imu = [0, 0, 0]

        self.output = []

    def recvProc(self):
        while self.viewer.is_alive:
            qposcontents, = self.sub.recv_multipart()
            self.data.qpos = np.frombuffer(qposcontents, np.float64)

            self.output.append(np.frombuffer(qposcontents, np.float64))

    def drawForce(self, v, p, color='r'):
        quat = np.zeros(4)
        mat = np.zeros(9)
        quat1 = np.zeros(4)
        mat1 = np.zeros(9)
        len = np.linalg.norm(v)
        dif = v / len
        scl = 2.0

        if color == 'r':
            rgb = np.array([1, 0, 0, 1])
        if color == 'g':
            rgb = np.array([0, 1, 0, 1])
        if color == 'b':
            rgb = np.array([0, 0, 1, 1])

        mujoco.mju_quatZ2Vec(quat, dif)
        mujoco.mju_quat2Mat(mat, quat)
        mujoco.mju_quatZ2Vec(quat1, -dif)
        mujoco.mju_quat2Mat(mat1, quat1)
        if ENABLE_VIEWER:
            # self.viewer.add_marker(pos=p, mat=-mat, size=np.array([0.005, 0.005, 2 * len]), type=mujoco.mjtGeom.mjGEOM_ARROW, rgba=rgb)
            self.viewer.add_marker(pos=p, mat=mat1, size=np.array([0.005, 0.005, len]), type=mujoco.mjtGeom.mjGEOM_LINE, rgba=rgb)

    def drawPoint(self, p, mat, color='r'):
        if ENABLE_VIEWER:
            if color == 'r':
                rgb = np.array([1, 0, 0, 1])
            if color == 'g':
                rgb = np.array([0, 1, 0, 1])
            if color == 'b':
                rgb = np.array([0, 0, 1, 1])
            self.viewer.add_marker(pos=p, mat=mat, size=np.array([0.005, 0.01, 0.02]), type=mujoco.mjtGeom.mjGEOM_BOX, rgba=rgb)
            # viewer.add_marker(pos=p, mat=mat, size=np.array([0.1, 0.1, 0.1]), type=mujoco.mjtGeom.mjGEOM_SPHERE, rgba=rgb)

    def run(self):
        loop_count = 0
        while self.viewer.is_alive:
            # r_thigh_imu_position = np.array(self.position_dict[0:3])
            # r_shank_imu_position = np.array(self.position_dict[3:6])
            # l_thigh_imu_position = np.array(self.position_dict[6:9])
            # l_shank_imu_position = np.array(self.position_dict[9:12])
            # r_arm_imu_position = np.array(self.position_dict[12:15])
            # r_forearm_imu_position = np.array(self.position_dict[15:18])
            # l_arm_imu_position = np.array(self.position_dict[18:21])
            # l_forearm_imu_position = np.array(self.position_dict[21:24])
            # pelvis_imu_position = np.array(self.position_dict[24:27])
            # back_imu_position = np.array(self.position_dict[27:30])
            #
            # r_thigh_imu_posture = np.array(self.posture_dict[0:9]).reshape(3, 3)
            # r_shank_imu_posture = np.array(self.posture_dict[9:18]).reshape(3, 3)
            # l_thigh_imu_posture = np.array(self.posture_dict[18:27]).reshape(3, 3)
            # l_shank_imu_posture = np.array(self.posture_dict[27:36]).reshape(3, 3)
            # r_arm_imu_posture = np.array(self.posture_dict[36:45]).reshape(3, 3)
            # r_forearm_imu_posture = np.array(self.posture_dict[45:54]).reshape(3, 3)
            # l_arm_imu_posture = np.array(self.posture_dict[54:63]).reshape(3, 3)
            # l_forearm_imu_posture = np.array(self.posture_dict[63:72]).reshape(3, 3)
            # pelvis_imu_posture = np.array(self.posture_dict[72:81]).reshape(3, 3)
            # back_imu_posture = np.array(self.posture_dict[81:90]).reshape(3, 3)

            # self.point_pelvis_imu = self.data.body('pelvis').xpos + pelvis_imu_position
            # self.drawForce(-pelvis_imu_position, self.data.body('pelvis').xpos, color='g')
            # self.drawPoint(self.point_pelvis_imu, pelvis_imu_posture, color='b')
            #
            # self.point_r_thigh_imu = self.data.body('femur_r').xpos + r_thigh_imu_position
            # self.drawForce(-r_thigh_imu_position, self.data.body('femur_r').xpos, color='g')
            # self.drawPoint(self.point_r_thigh_imu, r_thigh_imu_posture, color='b')
            #
            # self.point_r_shank_imu = self.data.body('tibia_r').xpos + r_shank_imu_position
            # self.drawForce(-r_shank_imu_position, self.data.body('tibia_r').xpos, color='g')
            # self.drawPoint(self.point_r_shank_imu, r_shank_imu_posture, color='b')
            #
            # self.point_l_thigh_imu = self.data.body('femur_l').xpos + l_thigh_imu_position
            # self.drawForce(-l_thigh_imu_position, self.data.body('femur_l').xpos, color='g')
            # self.drawPoint(self.point_l_thigh_imu, l_thigh_imu_posture, color='b')
            #
            # self.point_l_shank_imu = self.data.body('tibia_l').xpos + l_shank_imu_position
            # self.drawForce(-l_shank_imu_position, self.data.body('tibia_l').xpos, color='g')
            # self.drawPoint(self.point_l_shank_imu, l_shank_imu_posture, color='b')
            #
            # self.point_back_imu = self.data.body('torso').xpos + back_imu_position
            # self.drawForce(-back_imu_position, self.data.body('torso').xpos, color='g')
            # self.drawPoint(self.point_back_imu, back_imu_posture, color='b')
            #
            # self.point_r_arm_imu = self.data.body('humerus_r').xpos + r_arm_imu_position
            # self.drawForce(-r_arm_imu_position, self.data.body('humerus_r').xpos, color='g')
            # self.drawPoint(self.point_r_arm_imu, r_arm_imu_posture, color='b')
            #
            # self.point_r_forearm_imu = self.data.body('ulna_r').xpos + r_forearm_imu_position
            # self.drawForce(-r_forearm_imu_position, self.data.body('ulna_r').xpos, color='g')
            # self.drawPoint(self.point_r_forearm_imu, r_forearm_imu_posture, color='b')
            #
            # self.point_l_arm_imu = self.data.body('humerus_l').xpos + l_arm_imu_position
            # self.drawForce(-l_arm_imu_position, self.data.body('humerus_l').xpos, color='g')
            # self.drawPoint(self.point_l_arm_imu, l_arm_imu_posture, color='b')
            #
            # self.point_l_forearm_imu = self.data.body('ulna_l').xpos + l_forearm_imu_position
            # self.drawForce(-l_forearm_imu_position, self.data.body('ulna_l').xpos, color='g')
            # self.drawPoint(self.point_l_forearm_imu, l_forearm_imu_posture, color='b')

            loop_count += 1

            mujoco.mj_forward(self.model, self.data)
            if ENABLE_VIEWER:
                if self.viewer.is_alive:
                    self.viewer.render()


if __name__ == '__main__':
    client = walking_sub()
    time.sleep(1)
    client.run()

    with open('./data' + str(datetime.datetime.now().strftime('%Y年%m月%d日%H时%M分%S秒%f')[:-3]) + '.csv', 'w', newline='') as file:
        # with open('./data' + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(client.output)
    print('finish!')
