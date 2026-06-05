# IMU RS-485 链式轮询通信与 Wire485_lzd 上位机解析说明

本文根据以下内容整理：

- 固件文件：`Simple_MPU9250_with_Client3d_stm32f103.ino`
- 通信总结：`485通信方式.txt`
- 上位机代码：[Wire485_lzd/process.py](../Wire485_lzd/process.py)
- 简化读取类：[Wire485_lzd/imu_485_usb_new.py](../Wire485_lzd/imu_485_usb_new.py)

本文只解释通信机制、帧结构和 `process.py` 中的读取解析逻辑，不展开后续人体关节解算公式。

## 1. 系统拓扑

这套 485 IMU 系统不是“一个 IMU 一个串口”，而是“多个 IMU 共用一条 RS-485 半双工总线”。

当前 [Wire485_lzd/process.py](../Wire485_lzd/process.py) 中使用两条树莓派串口：

```python
self.ser_right = serial.Serial('/dev/ttySC0', 921600, timeout=0.01)
self.ser_left = serial.Serial('/dev/ttySC1', 921600, timeout=0.01)
```

含义可以理解为：

```text
右侧 6 个 IMU -> 同一条 485 总线 -> /dev/ttySC0
左侧 6 个 IMU -> 同一条 485 总线 -> /dev/ttySC1
```

代码里每条总线设置：

```python
self.imu_num = 6
self.txData = b'\xFF\xFF\x80\n'
self.imu_id = 0x80 + self.imu_num
```

也就是每条总线从 ID `0x80` 的 IMU 开始触发，期望经过 6 个 IMU 后，最后一个帧尾给出 `0x86`。

右侧总线解析后映射为：

```text
0 -> R_thigh
1 -> R_shank
2 -> R_foot
3 -> R_arm
4 -> R_forearm
5 -> Pelvis
```

左侧总线解析后映射为：

```text
0 -> L_thigh
1 -> L_shank
2 -> L_foot
3 -> L_arm
4 -> L_forearm
5 -> Back
```

## 2. 和蓝牙 IMU 轮询的根本区别

蓝牙 LPMS 方案中，每个 IMU 对应一个独立 RFCOMM 虚拟串口：

```text
IMU0 -> /dev/rfcomm0
IMU1 -> /dev/rfcomm1
...
IMU6 -> /dev/rfcomm6
```

Python 轮询蓝牙 IMU 时，是逐个检查多个独立串口。每个串口都有自己的内核缓冲区和应用层缓存，因此不同 IMU 的字节流天然隔离。

485 方案不同。每条 485 总线上有多个 IMU，共享同一对差分线，最终进入同一个串口设备：

```text
IMU0 \
IMU1  \
IMU2   -> RS-485 总线 -> /dev/ttySC0
...
IMU5  /
```

所以 485 方案不能依赖“每个 IMU 一个串口”来避免混流。RS-485 物理层本身也不会帮多个节点做冲突仲裁。如果多个 IMU 同时驱动总线发送，数据会冲突、乱码。

这套系统避免冲突的关键，不是硬件自动仲裁，而是固件实现的“链式轮询 / 级联接力 / 令牌传递”协议。

## 3. 链式轮询协议的核心思想

上位机只发送一次起始命令：

```text
FF FF 80 0A
```

所有 IMU 都能听到这条命令，但只有 `id_self == 0x80` 的 IMU 会响应。

第一个 IMU 发送自己的 38 字节数据帧，并在帧尾放入下一个 IMU 的触发命令：

```text
[IMU0 的 34 字节数据] FF FF 81 0A
```

这个帧尾同时会被总线上的所有 IMU 听到。只有 `id_self == 0x81` 的 IMU 会响应，于是第二个 IMU 发送：

```text
[IMU1 的 34 字节数据] FF FF 82 0A
```

依次接力，直到第 6 个 IMU 发送：

```text
[IMU5 的 34 字节数据] FF FF 86 0A
```

因此一轮完整采集为：

```text
上位机 -> IMU0 -> IMU1 -> IMU2 -> IMU3 -> IMU4 -> IMU5
```

不是：

```text
IMU0、IMU1、IMU2 同时发送
```

这就是多个 IMU 挂在同一条 485 总线上却不打架的核心原因。

## 4. IMU ID 配置

固件中通过条件编译选择每个 IMU 的 ID。例如：

```cpp
#define ID_130
```

对应：

```cpp
#ifdef ID_130
char id_self = 130;
#define OFFSETS ...
#endif
```

一条总线上 6 个 IMU 通常应配置为连续 ID：

```text
第 1 个 IMU: 0x80
第 2 个 IMU: 0x81
第 3 个 IMU: 0x82
第 4 个 IMU: 0x83
第 5 个 IMU: 0x84
第 6 个 IMU: 0x85
```

注意事项：

- ID 必须唯一，不能有两个 IMU 使用同一个 ID。
- ID 必须连续，因为每个 IMU 用 `id_self + 1` 触发下一个 IMU。
- 上位机的 `imu_num` 必须和实际总线上的 IMU 数一致。
- 最后一个 IMU 的 `id_self + 1` 不应误触发总线上额外设备。

如果两个 IMU ID 重复，例如两个设备都是 `0x82`，当总线上出现 `FF FF 82 0A` 时两个设备会同时发送，RS-485 总线就会冲突。

如果 ID 不连续，例如缺少 `0x82`，链式接力会在 `FF FF 82 0A` 处中断，后续 IMU 不会被触发。

## 5. 固件如何判断自己被触发

固件在 `loop()` 中持续读取串口：

```cpp
while (Serial.available())
{
    ret = Serial.read();
    if ((ret == 0x0a) &&
        (id_read_0 == id_self) &&
        (id_read_1 == 0xFF) &&
        (id_read_2 == 0xFF))
    {
        // 发送本 IMU 数据
    }

    id_read_2 = id_read_1;
    id_read_1 = id_read_0;
    id_read_0 = ret;
}
```

这相当于维护最近收到的 4 个字节。如果最近 4 个字节是：

```text
FF FF id_self 0A
```

当前 IMU 就认为自己被点名，需要发送数据。

由于总线上所有 IMU 都能听到所有字节，但每个 IMU 只响应自己的 ID，所以同一时刻理论上只有当前被点名的那个 IMU 会发送。

## 6. RS-485 半双工方向控制

RS-485 半双工总线同一时刻通常只能有一个设备发送。固件中使用 PB0 控制 485 收发器的发送使能：

```cpp
pinMode(PB0, OUTPUT);
digitalWrite(PB0, 0);
```

默认 PB0 为 0，IMU 处于接收模式。

当 IMU 被触发后：

```cpp
digitalWrite(PB0, 1);
Serial.write(buf, 38);
Serial.flush();
digitalWrite(PB0, 0);
```

含义是：

```text
PB0 = 1 -> 打开发送器
Serial.write(buf, 38) -> 发送 38 字节
Serial.flush() -> 等待串口数据真正发完
PB0 = 0 -> 关闭发送器，回到接收模式
```

`Serial.flush()` 很重要。如果还没等最后几个字节真正发出就关闭发送器，帧尾 `FF FF next_id 0A` 可能丢失，后面的 IMU 就无法被触发。

## 7. 单个 IMU 的 38 字节帧格式

每个 IMU 被触发后发送固定 38 字节：

```text
0  - 15: 四元数，4 个 float32，顺序 q.w, q.x, q.y, q.z
16 - 21: 陀螺仪，3 个 int16，gyro_x, gyro_y, gyro_z
22 - 27: 原始加速度，3 个 int16，acc_raw_x, acc_raw_y, acc_raw_z
28 - 33: 线加速度，3 个 int16，acc_real_x, acc_real_y, acc_real_z
34 - 37: 帧尾，同时是下一个 IMU 的触发命令
```

帧尾结构为：

```text
byte 34: 0xFF
byte 35: 0xFF
byte 36: id_self + 1
byte 37: 0x0A
```

固件中对应写法：

```cpp
memcpy(&buf[0],  &(q.w),        4);
memcpy(&buf[4],  &(q.x),        4);
memcpy(&buf[8],  &(q.y),        4);
memcpy(&buf[12], &(q.z),        4);

memcpy(&buf[16], &my_gyroo[0],  2);
memcpy(&buf[18], &my_gyroo[1],  2);
memcpy(&buf[20], &my_gyroo[2],  2);

memcpy(&buf[22], &my_accel[0],  2);
memcpy(&buf[24], &my_accel[1],  2);
memcpy(&buf[26], &my_accel[2],  2);

memcpy(&buf[28], &my_accel[3],  2);
memcpy(&buf[30], &my_accel[4],  2);
memcpy(&buf[32], &my_accel[5],  2);

buf[34] = 0xFF;
buf[35] = 0xFF;
buf[36] = id_self + 1;
buf[37] = '\n';

Serial.write(buf, 38);
```

因此每个 IMU 的 payload 数据为前 34 字节，最后 4 字节既是本帧帧尾，也是下一个 IMU 的触发命令。

## 8. 一条总线的一轮数据长度

如果每条总线有 6 个 IMU，每个 IMU 38 字节，那么一轮总长度为：

```text
readlen = 38 * 6 = 228 字节
```

[Wire485_lzd/process.py](../Wire485_lzd/process.py) 中正是这样读取：

```python
readlen = 38 * self.imu_num
data = self.ser_right.read(readlen)
```

左侧总线同理：

```python
data = self.ser_left.read(readlen)
```

这和蓝牙 LPMS 版非常不同：蓝牙版每个 IMU 独立到达、独立缓存、独立切帧；485 版则假设一次读取拿到一条总线上所有 IMU 级联返回的聚合包。

## 9. 上位机启动与循环触发

构造函数中，上位机打开两条串口后立即向两条总线写入起始触发命令：

```python
self.txData = b'\xFF\xFF\x80\n'
self.ser_right.write(self.txData)
self.ser_left.write(self.txData)
```

每次读完一轮后，再发送下一轮触发：

```python
self.ser_right.write(self.txData)
```

左侧同理：

```python
self.ser_left.write(self.txData)
```

所以每条 485 总线的循环是：

```text
上位机发送 FF FF 80 0A
  -> 总线上的 6 个 IMU 依次接力发送
  -> 上位机读取 38 * 6 字节
  -> 上位机解析并更新字典
  -> 上位机再次发送 FF FF 80 0A
```

主程序中依次读取右、左两条总线：

```python
read_imu.read_imu_data_right()
read_imu.read_imu_data_left()
```

这不是在一条总线上逐个查询 6 个 IMU，而是在两条总线上分别读取一个完整的 6-IMU 聚合包。

## 10. process.py 的完整性校验

`process.py` 当前做了两层检查。

第一层是长度检查：

```python
l = len(data)
if l == readlen:
    ...
else:
    print('err: received length=', l)
```

如果 `timeout=0.01` 内没有收满 228 字节，就会打印长度错误。

第二层是最后 ID 检查：

```python
id = data[l - 2]
if id == self.imu_id:
    ...
else:
    print('wrong id =', id)
```

其中：

```python
self.imu_id = 0x80 + self.imu_num
```

如果 `imu_num = 6`，期望最后一个帧尾 ID 是：

```text
0x80 + 6 = 0x86
```

最后一包帧尾应为：

```text
FF FF 86 0A
```

因此 `data[-2] == 0x86` 可以粗略说明这一轮链式接力走到了第 6 个 IMU。

更严格的校验方式是逐个检查每个 38 字节小包的帧尾：

```python
for k in range(self.imu_num):
    tail = data[k * 38 + 34 : k * 38 + 38]
    expected = bytes([0xFF, 0xFF, 0x80 + k + 1, 0x0A])
    if tail != expected:
        print("IMU", k, "tail error", tail, expected)
```

当前代码只检查最终 ID，如果中间某一包错位但最后字节偶然对上，可能不容易定位问题。调试总线稳定性时，建议加逐包帧尾检查。

## 11. process.py 的数据解析

每个 IMU 的有效数据为前 34 字节，所以代码按每 38 字节一个槽位切片：

```python
start_index = (i - 1) * 34 + (i - 1) * 4
end_index = start_index + 34
```

这个表达式等价于：

```python
start_index = (i - 1) * 38
end_index = start_index + 34
```

然后用：

```python
result = struct.unpack('4f 3h 3h 3h', data[start_index:end_index])
```

解析 34 字节：

```text
4f -> q.w, q.x, q.y, q.z
3h -> gyro_x, gyro_y, gyro_z
3h -> acc_raw_x, acc_raw_y, acc_raw_z
3h -> acc_real_x, acc_real_y, acc_real_z
```

解析后，代码把四元数从固件输出的 `(w, x, y, z)` 转成 Python 解算流程中使用的 `(x, y, z, w)`：

```python
q_raw = list(result[0:4])
q = [q_raw[1], q_raw[2], q_raw[3], q_raw[0]]
```

角速度缩放：

```python
g_raw = list(result[4:7])
g = [x / 938.734 for x in g_raw]
```

原始加速度缩放：

```python
a_raw = list(result[7:10])
a_raw = [a_raw[0] / 208.980, a_raw[1] / 208.980, a_raw[2] / 208.980]
```

虽然固件也发送了 `my_accel[3:6]` 作为线加速度，但当前 `process.py` 中没有直接使用 `result[10:13]`，而是用四元数旋转矩阵重新计算：

```python
Matrix = quaternion_to_matrix(q)
a_real = (a_raw - Matrix.T @ [0.0, 0.0, 9.8]).tolist()
```

最终每个 IMU 存入：

```python
imu_data = [q, g, a_raw, a_real]
```

并写入右侧或左侧字典：

```python
self.imu_data_dict_right[i - 1] = imu_data
self.imu_data_dict_left[i - 1] = imu_data
```

## 12. 485 方式为什么不会打架

从物理层看，RS-485 总线是共享的，如果两个 IMU 同时打开发送器，必然会冲突。

本系统避免冲突依赖下面几个条件：

1. 每个 IMU 有唯一 ID。
2. 每个 IMU 只响应 `FF FF 自己ID 0A`。
3. 每个 IMU 的帧尾触发下一个 IMU。
4. PB0 默认关闭发送器，只有被点名时短暂发送。
5. 发送后 `Serial.flush()` 等待数据真正发完，再关闭发送器。
6. ID 连续，使令牌按顺序传递。

所以总线上任意时刻只有一个“令牌持有者”发送数据。这个令牌就是：

```text
FF FF current_id 0A
```

前一个 IMU 的帧尾会产生下一个令牌：

```text
FF FF next_id 0A
```

这是一种软件协议层面的时序控制，不是 RS-485 硬件本身自动完成的仲裁。

## 13. 和蓝牙轮询方案的对比

| 项目 | 蓝牙 LPMS 方案 | 485 链式轮询方案 |
|---|---|---|
| 物理/链路结构 | 每个 IMU 一个 RFCOMM 虚拟串口 | 多个 IMU 共享一条 485 总线 |
| Linux 设备 | `/dev/rfcomm0..6` | `/dev/ttySC0`、`/dev/ttySC1` |
| 上位机读取 | 逐个串口轮询或 epoll 监听多个 fd | 每条总线一次读 `38 * imu_num` 字节 |
| IMU 发送方式 | 每个 IMU 独立链路上报 | 被触发后按 ID 接力发送 |
| 防冲突机制 | 链路天然隔离 | 唯一 ID + 帧尾触发下一个 ID |
| 缓存方式 | 每个串口独立 `buf_list[i]`，按帧头/长度/帧尾切帧 | 当前代码按固定长度读整包，无 per-IMU 缓存 |
| 主要风险 | 蓝牙掉线、rfcomm 重连、帧碎片 | ID 重复/不连续、总线错位、长度不足、线缆质量 |
| 优化方向 | epoll 监听多个串口 fd | 可加逐包帧尾校验、超时重触发、总线级 epoll/timerfd |

蓝牙方案是“多个独立数据流的轮询/epoll”；485 方案是“一个共享数据流上的链式时序协议”。

## 14. 当前 process.py 的风险与改进建议

当前 [Wire485_lzd/process.py](../Wire485_lzd/process.py) 能工作在一个重要假设上：每次 `read(38 * imu_num)` 都能从包起点开始，且在 timeout 内收满一轮完整数据。

实际运行中需要注意：

- 如果串口缓冲中残留半包，固定长度读取可能从中间开始，导致解析错位。
- 如果某个 IMU 未响应，链式接力中断，上位机只能读到不足 228 字节。
- 如果中间帧尾坏了，后续 IMU 不会被触发。
- 如果最后 ID 正确但中间数据错位，当前只检查 `data[-2]` 可能不够。
- 921600 波特率较高，对接线、终端电阻、共地、屏蔽和收发器质量要求较高。

建议后续调试或升级时优先增加：

1. 逐个 38 字节小包检查帧尾。
2. 发生长度错误时清空串口输入缓冲，再重新发送 `FF FF 80 0A`。
3. 统计右、左两条总线的错误率和耗时。
4. 记录每轮 `read()` 实际长度、最终 ID、每包 tail。
5. 若要做低延迟版本，可让两条总线分别由两个线程或 epoll/timerfd 管理，再把左右数据合并给解算线程。

## 15. 总结

这套 485 IMU 通信不是普通的“上位机逐个查询 6 个地址”，也不是“6 个 IMU 同时主动上报”。它是固件实现的链式轮询协议：

```text
上位机发 FF FF 80 0A
  -> ID 0x80 的 IMU 发送自己的 38 字节，并触发 0x81
  -> ID 0x81 的 IMU 发送自己的 38 字节，并触发 0x82
  -> ...
  -> ID 0x85 的 IMU 发送自己的 38 字节，并触发 0x86
  -> 上位机收到 38 * 6 字节，检查最后 ID 为 0x86
```

它不会打架的前提是：IMU ID 唯一且连续、PB0 方向控制正确、每个 IMU 只在收到自己的触发命令后发送，并且发送结束后回到接收模式。

[Wire485_lzd/process.py](../Wire485_lzd/process.py) 正是围绕这个协议写的上位机解析程序：两条串口分别对应左右两条 485 总线，每条总线一次读取 6 个 IMU 的聚合包，按固定 38 字节槽位拆包，解析四元数、角速度和加速度，然后送入后续姿态解算。
