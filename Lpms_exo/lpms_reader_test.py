#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time

import openzen  # OpenZen Python binding


def connect_by_mac(client, mac: str, baudrate: int = 115200, io_system: str = "Bluetooth"):
    # 经典蓝牙 B2：IO system = "Bluetooth"，name = MAC
    # BLE 模式则 io_system = "Ble"
    err, sensor = client.obtainSensorByName(io_system, mac, baudrate)
    if err:
        raise RuntimeError(f"obtainSensorByName failed: {err}")
    return sensor


def connect_by_scan(client, force_baudrate=None):
    err = client.listSensorsAsync()
    if err:
        raise RuntimeError(f"listSensorsAsync failed: {err}")

    sensors = []
    listing_complete = False
    t0 = time.time()

    while not listing_complete:
        success, event = client.waitForNextEvent()
        if not success:
            continue

        # client-level events have component.handle == 0
        if event.component.handle == 0:
            if event.eventType == openzen.ZenEventType_SensorFound:
                desc = event.data.sensorFound
                sensors.append(desc)
                print(f"[FOUND] name={desc.name} ioType={desc.ioType} serial={desc.serialNumber}")
            elif event.eventType == openzen.ZenEventType_SensorListingProgress:
                p = event.data.sensorListingProgress.progress
                c = event.data.sensorListingProgress.complete
                print(f"[LISTING] progress={p*100:.1f}% complete={c}")
                if c:
                    listing_complete = True

        if time.time() - t0 > 15:
            print("Listing timeout (15s)")
            break

    if not sensors:
        raise RuntimeError("No sensors found.")

    for i, s in enumerate(sensors):
        print(f"{i}: {s.name} ({s.ioType})")

    idx = int(input(f"Select index [0-{len(sensors)-1}]: ").strip())
    desc = sensors[idx]

    # 对 B2/BE/ME：经常需要 115200
    if force_baudrate is not None:
        desc.baudRate = int(force_baudrate)

    err, sensor = client.obtainSensor(desc)
    if err:
        raise RuntimeError(f"obtainSensor failed: {err}")
    return sensor


def main():
    # 参数：
    # 1) 直接连 MAC：python3 read_lpms_b2.py 00:04:3E:4B:32:95
    # 2) 扫描再选： python3 read_lpms_b2.py scan
    # 3) BLE：python3 read_lpms_b2.py ble 00:11:22:33:FF:EE
    mode = sys.argv[1].lower() if len(sys.argv) >= 2 else "scan"

    openzen.set_log_level(openzen.ZenLogLevel_Info)

    err, client = openzen.make_client()
    if err:
        raise RuntimeError(f"make_client failed: {err}")

    try:
        if mode == "scan":
            sensor = connect_by_scan(client, force_baudrate=115200)
        elif mode == "ble":
            mac = sys.argv[2]
            sensor = connect_by_mac(client, mac, baudrate=115200, io_system="Ble")
        else:
            mac = sys.argv[1]
            sensor = connect_by_mac(client, mac, baudrate=115200, io_system="Bluetooth")

        # 获取 IMU component
        has_imu, imu = sensor.getAnyComponentOfType(openzen.g_zenSensorType_Imu)
        if not has_imu:
            raise RuntimeError("Connected sensor has no IMU component")

        # 打开 streaming（关键）
        err = imu.setBoolProperty(openzen.ZenImuProperty_StreamData, True)
        if err:
            raise RuntimeError(f"Failed to enable streaming: {err}")

        print("Streaming... Ctrl+C to stop.")
        imu_handle = imu.component().handle

        while True:
            success, event = client.waitForNextEvent()
            if not success:
                continue

            if event.component.handle != imu_handle:
                continue

            if event.eventType == openzen.ZenEventType_ImuData:
                a = event.data.imuData.a
                g1 = event.data.imuData.g1
                print(
                    f"a[m/s^2]=({a[0]: .4f}, {a[1]: .4f}, {a[2]: .4f})  "
                    f"g1[deg/s]=({g1[0]: .4f}, {g1[1]: .4f}, {g1[2]: .4f})"
                )

    except KeyboardInterrupt:
        pass
    finally:
        try:
            sensor.release()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()