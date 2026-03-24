# import asyncio
# import bleak
# import wt_device_model
#
# TARGET_MAC = "D7:5E:B3:78:1E:62"   # 改成你那台 WT901 的 MAC
# BLEDevice = None
#
# async def scanByMac(device_mac):
#     global BLEDevice
#     print("Searching for Bluetooth device by MAC......")
#     BLEDevice = await bleak.BleakScanner.find_device_by_address(device_mac, timeout=20)
#
# def updateData(DeviceModel):
#     print(DeviceModel.deviceData)
#
# async def main():
#     await scanByMac(TARGET_MAC)
#     if BLEDevice is None:
#         print("This BLEDevice was not found!!")
#         return
#
#     device = wt_device_model.DeviceModel("WT901", BLEDevice, updateData)
#     await device.openDevice()
#
# if __name__ == '__main__':
#     asyncio.run(main())

import asyncio
from bleak import BleakScanner
import wt_device_model_hy

class WTMultiIMU:
    def __init__(self, mac_list):
        self.mac_list = mac_list
        self.imu_num = len(mac_list)

        # 数值结构：idx -> [quat_xyzw, gyro_xyz]
        self.imu_data_dict = {i: [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]] for i in range(self.imu_num)}

        self.devices = [None] * self.imu_num  # DeviceModel list

    def on_numeric(self, imu_idx, q_xyzw, g_xyz):
        # 数字列表
        self.imu_data_dict[imu_idx] = [q_xyzw, g_xyz]

        # 你要打印/发布都在这里做
        print("idx:", imu_idx, "data:", self.imu_data_dict[imu_idx])

    async def run_all(self):
        # 扫描一次,避免 InProgress
        devices = await BleakScanner.discover(timeout=8.0)
        by_addr = {d.address.upper(): d for d in devices}

        missing = [m for m in self.mac_list if m not in by_addr]
        if missing:
            raise RuntimeError(f"Not found in scan: {missing}")

        # 并发连接
        tasks = []
        for i, mac in enumerate(self.mac_list):
            dev = by_addr[mac]
            dm = wt_device_model_hy.DeviceModel(
                deviceName=f"WT901_{i}",
                BLEDevice=dev,
                callback_method=None,
                imu_idx=i,
                callback_numeric=self.on_numeric,
            )
            tasks.append(asyncio.create_task(dm.openDevice()))

        await asyncio.gather(*tasks)

async def main():
    # 把你要连的多个 IMU MAC 填在这里（顺序就是 imu_idx）
    mac_list = [
        "E6:4F:C8:40:5E:BB",
        "D7:5E:B3:78:1E:62",
        # "xx:xx:xx:xx:xx:xx",
    ]

    mgr = WTMultiIMU(mac_list)
    await mgr.run_all()

if __name__ == "__main__":
    asyncio.run(main())