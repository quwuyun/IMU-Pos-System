import serial
import time
import struct

# port = '/dev/rfcomm1'
port = 'COM14'
baud = 921600


def extract_frames(buf):
    """从缓存中提取所有完整的帧，返回 (完整帧列表, 剩余缓存)"""
    frames = []
    while True:
        start = buf.find(b'\x3A')  # 找包头
        if start < 0:
            break
        if len(buf) < start + 7:
            break

        data_len = int.from_bytes(buf[start + 5:start + 7], 'little')
        frame_len = 7 + data_len + 4  # header+payload+LRC2+tail2

        if len(buf) < start + frame_len:
            break

        frame = buf[start:start + frame_len]

        # 验证尾部
        if frame.endswith(b'\x0D\x0A'):
            frames.append(frame)
            buf = buf[start + frame_len:]
        else:
            buf = buf[start + 1:]
    return frames, buf


def parse_lpms_payload(payload_bytes):
    """解析 LPMS‑B2 payload"""

    data = {}
    if len(payload_bytes) < 2:
        return data

    # ---- 时间戳（前两字节，小端 uint16，单位 tick -> 秒）----
    timestamp_raw = int.from_bytes(payload_bytes[0:2], 'little')
    data["timestamp"] = timestamp_raw / 400.0  # 每个 tick = 1/400 秒
    # ---- 其它为 float32，小端 ----
    floats = []
    for i in range(4, len(payload_bytes), 4):
        if i + 4 <= len(payload_bytes):
            val = struct.unpack('<f', payload_bytes[i:i + 4])[0]
            floats.append(val)
    try:
        # 88字节
        data["gyro"] = tuple(floats[0:3])
        data["acc"] = tuple(floats[3:6])
        data["mag"] = tuple(floats[6:9])
        data["quat"] = tuple(floats[9:13])
        data["euler"] = tuple(floats[13:16])
        data["lin_acc"] = tuple(floats[16:19])
    except Exception as e:
        print("⚠️ 字段解析长度异常:", e)
    return data


try:
    ser = serial.Serial(port, baud, timeout=0.1)
    print(f"✅ 连接到 {port}，开始读取 LPMS‑B2 数据流，按 Ctrl+C 退出。\n")

    buffer = b""
    count = 0

    while True:
        chunk = ser.read(103)
        if chunk:
            buffer += chunk
            frames, buffer = extract_frames(buffer)
            for frame in frames:
                # print(frame.hex(" "))
                count += 1
                data_len = int.from_bytes(frame[5:7], 'little')
                payload = frame[7:7 + data_len]
                # print(payload.hex(" "))
                parsed = parse_lpms_payload(payload)

                print(f"\n--- Frame #{count} | {len(frame)} bytes | payload={data_len} ---")
                print(f"Timestamp: {parsed.get('timestamp', 0):.3f}")
                print("data:", parsed)
                if parsed.get("gyro"):
                    gx, gy, gz = parsed["gyro"]
                    print(f"Gyro [rad/s]: {gx:+.3f}, {gy:+.3f}, {gz:+.3f}")
                if parsed.get("acc"):
                    ax, ay, az = parsed["acc"]
                    print(f"Acc  [g]: {ax:+.3f}, {ay:+.3f}, {az:+.3f}")
                if parsed.get("euler"):
                    r, p, y = parsed["euler"]
                    print(f"Euler [rad]: Roll={r:+.3f}, Pitch={p:+.3f}, Yaw={y:+.3f}")
                if parsed.get("quat"):
                    w, x, y, z = parsed["quat"]
                    print(f"quat : w={w:+.3f}, x={x:+.3f}, y={y:+.3f}, z={z:+.3f}")
                if parsed.get("lin_acc"):
                    lx, ly, lz = parsed["lin_acc"]
                    print(f"LinAcc[g]: {lx:+.3f}, {ly:+.3f}, {lz:+.3f}")

        time.sleep(0.02)
        # print("Buffered bytes:", ser.in_waiting)

except KeyboardInterrupt:
    print("\n👋 退出。")
except Exception as e:
    print("❌ 连接失败:", e)