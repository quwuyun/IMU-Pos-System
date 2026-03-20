"""
连不上三步操作：

1.查找使用的蓝牙设备，并清空BlueZ数据库

sudo systemctl stop bluetooth
sudo rm -rf /var/lib/bluetooth/*
sudo systemctl start bluetooth

2.重置IMU出厂设置；重启树莓派；重置蓝牙适配器；
sudo hciconfig hci0 reset

3.重新配对（不要scan on）
power on
agent KeyboardDisplay
default-agent
pairable on
discoverable on
pair <MAC>  # 必须出现pair的密钥请求yes or no的选项，选择yes则成功配对
trust <MAC>
connect <MAC>  # 没连上不影响
"""

import serial
import struct
import time

print("test")

port = '/dev/rfcomm4'
# port = 'COM14'
baud = 115200

try:
    ser = serial.Serial(port, baud, timeout=1)
    print(f"✅ 连接到 {port}，开始读取 LPMS‑B2 数据流，按 Ctrl+C 退出。\n")

    while True:
        data = ser.read(256)
        if data:
            # 仅展示前 16 字节的十六进制
            # print(" ".join(f"{b:02X}" for b in data[:32]))
            print(" ".join(f"{b:02X}" for b in data[:]))
        time.sleep(0.05)
except KeyboardInterrupt:
    print("退出。")
except Exception as e:
    print("❌ 连接失败:", e)
