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
