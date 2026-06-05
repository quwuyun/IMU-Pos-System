# 蓝牙IMU系统 工作流与控制设计

本文面向 [process_pi_parallel.py](../process_pi_parallel.py)，解释它如何用 `epoll` 读取多个 LPMS-B2 IMU、如何管理串口缓冲、如何把数据交给后台线程，以及后续如果要接入“关节数据 -> 神经网络推理 -> 电机驱动”，应该怎样组织 `epoll`、线程和队列。

本文不展开 `imu_data_solving()` 内部的姿态解算和关节数学，只说明它在系统中的位置、输入输出和调度关系。



# 第一部分：process_pi_parallel 工作流程

## 1. 总体结构

`process_pi.py` 的原始模式是：

```text
主循环
  -> 顺序轮询 rfcomm0
  -> 顺序轮询 rfcomm1
  -> ...
  -> 顺序轮询 rfcomm6
  -> 标定或解算
  -> ZMQ 发布
```

`process_pi_parallel.py` 的当前模式是：

```text
主线程：IMU I/O
  -> epoll 只返回有数据或出错的串口 fd
  -> 对就绪 fd 读取字节
  -> 从对应 IMU 缓冲中切出完整帧
  -> 解析出 gyro 和 quat
  -> 更新 imu_data_dict
  -> 将最新 IMU 快照投递给 solve_queue

后台线程：solver_thread
  -> 从 solve_queue 取最新 IMU 快照
  -> 判断所有 IMU 是否已经产生有效数据
  -> 完成标定流程
  -> 调用 imu_data_solving()
  -> 通过 ZMQ 发布 data.qpos
```

核心思想是两点：

1. I/O 使用 `epoll` 事件驱动，不再每轮无差别检查所有串口。
2. 主线程只负责读数据和投递快照，标定、解算、发布放到后台线程，降低主 I/O 路径的周期和抖动。

## 2. 关键数据结构

### 2.1 串口与缓冲

```python
self.ser_list = []
self.buf_list = []
```

`ser_list[i]` 保存第 `i` 个 IMU 对应的 `serial.Serial` 对象。代码中的端口顺序来自：

```python
IMU_PORTS = [port0, port1, port2, port3, port4, port5, port6]
```

也就是 `/dev/rfcomm0` 到 `/dev/rfcomm6`。

`buf_list[i]` 是第 `i` 个 IMU 的字节缓存。它非常关键，因为串口读取并不保证“一次 read 正好读到一帧”。可能出现三种情况：

- 只读到半帧。
- 一次读到多帧。
- 前面残留了一些无效字节，新帧从中间某个位置开始。

所以每个 IMU 都必须有独立缓存，不能把多个 IMU 的字节混在一起处理。

### 2.2 最新 IMU 数据字典

```python
self.imu_data_dict[i] = [[0, 0, 0, 1], [0, 0, 0]]
```

每个条目的格式是：

```text
imu_data_dict[i] = [quat, gyro]
quat = [x, y, z, w]
gyro = [gx, gy, gz]
```

代码从 LPMS payload 中读出的四元数原始顺序是 `(w, x, y, z)`，之后转换成 scipy 使用的 `(x, y, z, w)`：

```python
q = parsed_data["quat"]
q = [q[1], q[2], q[3], q[0]]
```

当前解包时默认 IMU 与身体部位的顺序是：

```text
0 -> Pelvis
1 -> R_thigh
2 -> L_thigh
3 -> R_shank
4 -> L_shank
5 -> R_foot
6 -> L_foot
```

这一点后续接神经网络时也要保持一致，否则网络输入的关节语义会错位。

### 2.3 last_seen

```python
self.last_seen = [0.0 for _ in range(self.imu_num)]
```

`last_seen[i]` 记录第 `i` 个 IMU 最近一次成功解析出合法 `gyro` 和 `quat` 的系统时间。

它不是“串口已经打开”的标记，而是“应用层已经收到并解析出有效数据”的标记。这个区别很重要：蓝牙连接成功、`/dev/rfcommN` 存在、呼吸灯正常，都不等于代码已经拿到了完整帧。

当前用途：

- 标定前判断所有 IMU 是否都已经有新鲜数据。
- 重连时把对应 IMU 的 `last_seen` 清零，避免用旧状态误判它已经可用。

### 2.4 epoll 与 fd_map

```python
self.epoll = select.epoll()
self.fd_map = {}  # fd -> imu_idx
```

`epoll` 只能告诉程序“哪个文件描述符有事件”，它不会直接告诉你这是第几个 IMU。所以代码维护了一个反向映射：

```text
fd_map[fd] = imu_idx
```

初始化时，每个串口对象都通过 `ser.fileno()` 取到底层 fd，然后注册到 `epoll`：

```python
fd = ser.fileno()
self.fd_map[fd] = i
self.epoll.register(fd, select.EPOLLIN)
```

当某个 fd 可读时，主线程通过 `fd_map` 找回对应 IMU 编号，再读取对应 `ser_list[imu_idx]`。

### 2.5 solve_queue

```python
self.solve_queue = queue.Queue(maxsize=1)
```

这是主线程和解算线程之间的交接点。`maxsize=1` 是实时系统里很有用的设计：

- 如果解算线程处理慢了，不让旧数据无限排队。
- 队列满时先丢掉旧快照，再放入新快照。
- 解算线程永远尽量处理最新姿态，而不是补处理过期帧。

当前投递方式：

```python
if self.solve_queue.full():
    _ = self.solve_queue.get_nowait()
self.solve_queue.put_nowait(copy.deepcopy(self.imu_data_dict))
```

这里使用 `deepcopy` 的目的是把当前 IMU 字典复制成一个快照，避免主线程继续修改 `imu_data_dict` 时影响后台线程正在处理的数据。

## 3. 启动流程

程序入口在文件末尾：

```python
if __name__ == '__main__':
    print('process parallel init')
    IMU_PORTS = [port0, port1, port2, port3, port4, port5, port6]
    read_imu = imu_posture(ports=IMU_PORTS)
    while True:
        read_imu.read_imu_data()
```

创建 `imu_posture` 对象时，初始化顺序如下。

### 3.1 打开所有 IMU 串口

构造函数遍历传入的 `ports`：

```python
ser = serial.Serial(port, baudrate, timeout=0.01)
ser.reset_input_buffer()
self.ser_list.append(ser)
self.buf_list.append(b"")
```

这里做了三件事：

1. 打开 `/dev/rfcommN`。
2. 清空输入缓冲，尽量丢掉旧连接残留的字节。
3. 为该 IMU 建立独立的空缓存。

### 3.2 初始化默认数据

每个 IMU 的默认数据都是单位四元数和零角速度：

```text
quat = [0, 0, 0, 1]
gyro = [0, 0, 0]
```

这样即便某些 IMU 还没有首帧，数据结构也是完整的。但标定并不会仅仅因为有默认值就开始，后面还会用 `last_seen` 判断有效首帧是否到达。

### 3.3 加载标定矩阵与 MuJoCo 数据容器

构造函数读取 `calibration/rot_*.txt`，并加载 MuJoCo 模型：

```python
self.model_file = './model/new/walk_new_quat_body0901.xml'
self.model = mujoco.MjModel.from_xml_path(filename=self.model_file)
self.data = mujoco.MjData(self.model)
```

这些变量主要服务于后续标定和 `imu_data_solving()`。本文不展开解算公式，只需要知道：解算结果会写入 `self.data.qpos` 和 `self.data.qvel`。

### 3.4 创建 ZMQ 发布器

并行版把 ZMQ publisher 放进 `imu_posture` 内部，由后台解算线程发布：

```python
self.context = zmq.Context()
self.publisher = self.context.socket(zmq.PUB)
self.publisher.bind("tcp://*:5555")
```

当前发布内容是：

```python
self.publisher.send_multipart([self.data.qpos.tobytes()])
```

也就是只发布 `qpos` 的原始 bytes。

### 3.5 启动后台解算线程

```python
self.solver_thread = threading.Thread(target=self._solver_loop, daemon=True)
self.solver_thread.start()
```

`daemon=True` 表示主程序退出时该线程不会阻止进程结束。它一直在 `_solver_loop()` 中等待 `solve_queue` 的最新 IMU 快照。

### 3.6 注册 epoll

最后，构造函数创建 `epoll`，并把所有串口 fd 注册进去：

```python
self.epoll = select.epoll()
self.fd_map = {}
for i, ser in enumerate(self.ser_list):
    fd = ser.fileno()
    self.fd_map[fd] = i
    self.epoll.register(fd, select.EPOLLIN)
```

之后主线程不需要每次遍历所有串口来检查谁有数据，直接问内核：

```text
现在有哪些 fd 可读？
```

这就是 `epoll` 降低 I/O 周期的核心。

## 4. 主循环工作流

主循环只做一件事：

```python
read_imu.read_imu_data()
```

`read_imu_data()` 是并行版的 I/O 核心。

### 4.1 记录 I/O 起始时间

```python
t0 = time.perf_counter()
```

这是为了统计纯 I/O 耗时。这个耗时包括：

- `epoll.poll()`。
- 对就绪 fd 的读取。
- 帧提取与 payload 解析。
- 投递快照前的一些轻量逻辑。

它不包括 `imu_data_solving()` 的数学解算，因为解算在后台线程。

### 4.2 非阻塞 epoll.poll

```python
events = self.epoll.poll(0)
```

参数 `0` 表示非阻塞：

- 有事件就立刻返回事件列表。
- 没有事件就立刻返回空列表。

这会把主循环延迟压得很低，但 CPU 占用可能更高。如果将来希望降低 CPU 占用，可以改成非常小的超时，例如 `0.001` 到 `0.005` 秒。代价是最坏情况下会引入对应等待延迟。

### 4.3 处理 epoll 事件

事件处理逻辑是：

```python
for fd, event in events:
    if event & (select.EPOLLIN | select.EPOLLPRI):
        self._handle_fd_event(fd)
    elif event & (select.EPOLLHUP | select.EPOLLERR):
        self._reopen(imu_idx)
```

含义如下：

- `EPOLLIN`：普通可读事件，说明该串口有数据可以读。
- `EPOLLPRI`：高优先级可读事件，这里也按可读处理。
- `EPOLLHUP`：设备挂起，常见于蓝牙断开或 fd 状态异常。
- `EPOLLERR`：fd 出错，也进入重连流程。

### 4.4 记录 I/O 耗时

```python
t1 = time.perf_counter()
io_elapsed_ms = (t1 - t0) * 1000.0
self.cycle_times_io.append(io_elapsed_ms)
```

这组数据用于观察 epoll I/O 路径是否稳定。它和主循环外层的 `cycle_times` 不完全一样：

- `cycle_times_io` 是 `read_imu_data()` 内部记录的 I/O 时间。
- `cycle_times` 是入口主循环包住 `read_imu_data()` 的时间。
- `cycle_times_solve` 是后台线程中标定或解算实际执行的时间。

### 4.5 投递最新 IMU 快照

主线程每次执行完 I/O 后都会尝试把当前 `imu_data_dict` 投递给解算线程：

```python
if self.solve_queue.full():
    _ = self.solve_queue.get_nowait()
self.solve_queue.put_nowait(copy.deepcopy(self.imu_data_dict))
```

这意味着即便本轮没有新的串口事件，也可能投递一次当前最新状态。这样后台线程可以按主循环频率推进标定或解算；如果希望进一步减少无效投递，后续可以加一个 `updated` 标志，只在至少一个 IMU 更新时投递。

## 5. 单个 fd 的读取与解析

当某个串口 fd 可读时，`read_imu_data()` 调用：

```python
self._handle_fd_event(fd)
```

### 5.1 fd 找回 IMU 编号

```python
imu_idx = self.fd_map.get(fd)
ser = self.ser_list[imu_idx]
```

`epoll` 返回的是 fd，不是端口名，所以要通过 `fd_map` 找到 IMU 编号，再去 `ser_list` 中取串口对象。

### 5.2 读取当前内核缓冲中已有的字节

```python
n = ser.in_waiting
if n > 0:
    chunk = ser.read(n)
```

`in_waiting` 表示 pyserial 当前知道的待读字节数。这里一次读走已有字节，避免一字节一字节调用 `read()` 带来的 syscall 开销。

### 5.3 追加到对应 IMU 的缓存

```python
buf = self.buf_list[imu_idx] + chunk
frames, buf = self.extract_frames(buf)
self.buf_list[imu_idx] = buf
```

新读到的 `chunk` 不直接解析，而是先追加到该 IMU 的历史残留缓存中。`extract_frames()` 返回两部分：

- `frames`：已经切出来的完整帧列表。
- `buf`：剩下的不完整字节，留到下次继续拼。

### 5.4 只使用最新完整帧

```python
if frames:
    latest_frame = frames[-1]
```

如果一次读到了多帧，代码只解析最后一帧。实时控制场景里，这通常是合理的：宁可丢弃旧姿态，也不要排队处理过期数据。

### 5.5 解析 payload 并更新最新数据

```python
data_len = int.from_bytes(latest_frame[5:7], 'little')
payload = latest_frame[7:7 + data_len]
parsed_data = self.parse_lpms_payload(payload)
```

如果解析结果同时包含 `gyro` 和 `quat`：

```python
self.imu_data_dict[imu_idx] = [q, g]
self.last_seen[imu_idx] = time.time()
```

这一步完成后，该 IMU 的最新姿态和角速度才算真正可用。

## 6. 帧提取逻辑

`extract_frames(buf)` 负责从单个 IMU 的缓存中提取完整 LPMS 帧。

### 6.1 查找帧头

```python
start = buf.find(b'\x3A')
```

LPMS 帧头是 `0x3A`。如果缓存中找不到帧头，说明当前没有可解析帧。

### 6.2 检查 header 是否足够

```python
if len(buf) < start + 7:
    break
```

代码至少需要 7 字节 header 才能读出 payload 长度。如果不够，说明可能是半帧，保留缓存等待下一次 read。

### 6.3 读取 payload 长度

```python
data_len = int.from_bytes(buf[start + 5:start + 7], 'little')
frame_len = 7 + data_len + 4
```

帧总长度由三部分组成：

```text
header 7 字节 + payload data_len 字节 + LRC 2 字节 + tail 2 字节
```

所以代码写成 `7 + data_len + 4`。

### 6.4 检查完整帧是否到齐

```python
if len(buf) < start + frame_len:
    break
```

如果不够一整帧，保留缓存等待下次读取。

### 6.5 校验帧尾

```python
if frame.endswith(b'\x0D\x0A'):
    frames.append(frame)
    buf = buf[start + frame_len:]
else:
    buf = buf[start + 1:]
```

合法帧尾是 CR LF，即 `0x0D 0x0A`。

如果帧尾不对，代码不会把它当成完整帧，而是从当前帧头后移一字节继续搜索。这可以让解析逻辑从错位或残留垃圾字节中恢复。

## 7. payload 解析逻辑

`parse_lpms_payload(payload_bytes)` 负责把完整帧里的 payload 转成结构化数据。

### 7.1 时间戳

```python
timestamp_raw = int.from_bytes(payload_bytes[0:2], 'little')
data["timestamp"] = timestamp_raw / 100.0
```

前 2 字节按小端 `uint16` 解析，代码中除以 100 得到秒。

### 7.2 浮点数据

```python
for i in range(4, len(payload_bytes), 4):
    val = struct.unpack('<f', payload_bytes[i:i + 4])[0]
    floats.append(val)
```

从 payload offset 4 开始，每 4 字节按小端 float32 解析。

当前代码使用：

```python
data["gyro"] = tuple(floats[0:3])
data["quat"] = tuple(floats[3:7])
```

也就是：

```text
floats[0:3] -> gyro_x, gyro_y, gyro_z
floats[3:7] -> quat_w, quat_x, quat_y, quat_z
```

解析函数只负责把 payload 转成 `timestamp`、`gyro`、`quat`，不做姿态解算。

## 8. 重连流程

串口异常可能出现在两类地方：

1. `epoll` 返回 `EPOLLHUP` 或 `EPOLLERR`。
2. `_handle_fd_event()` 中 `ser.in_waiting` 或 `ser.read()` 抛出 `OSError` / `serial.SerialException`。

两种情况都会进入：

```python
self._reopen(imu_idx)
```

重连流程如下。

### 8.1 注销旧 fd

```python
old_fd = old.fileno()
self.epoll.unregister(old_fd)
del self.fd_map[old_fd]
```

这是 epoll 版本必须做的步骤。因为重新打开同一个 `/dev/rfcommN` 后，底层 fd 通常会变化。如果不注销旧 fd、不注册新 fd，后续 `epoll` 可能监听的是已经失效的 fd。

### 8.2 关闭旧串口

```python
old.close()
```

关闭旧的 `serial.Serial` 对象，释放系统资源。

### 8.3 清除 last_seen

```python
self.last_seen[imu_idx] = 0.0
```

重连期间，该 IMU 被认为暂时不可用。只有它重新收到并成功解析合法帧后，`last_seen` 才会再次更新。

### 8.4 重新打开端口并清缓存

```python
time.sleep(delay)
ser = serial.Serial(port, self.baudrate, timeout=0.01)
ser.reset_input_buffer()
self.ser_list[imu_idx] = ser
self.buf_list[imu_idx] = b""
```

重连会等待 `delay` 秒，默认 1 秒，给蓝牙 RFCOMM 一点恢复时间。

### 8.5 注册新 fd

```python
new_fd = ser.fileno()
self.fd_map[new_fd] = imu_idx
self.epoll.register(new_fd, select.EPOLLIN)
```

到这里，该 IMU 的新串口对象才重新进入 epoll 监听。

## 9. 后台线程的非解算流程

`_solver_loop()` 是后台线程入口。它的核心不是 I/O，而是从 `solve_queue` 取主线程投递的最新快照。

### 9.1 等待最新快照

```python
data = self.solve_queue.get(timeout=0.1)
```

如果 0.1 秒没有新数据，就继续等待。这里不会忙等。

### 9.2 将快照写回共享字典

```python
for k, v in data.items():
    self.imu_data_dict[k] = v
```

当前代码仍然复用了 `self.imu_data_dict` 作为解算输入。更理想的后续结构是让解算函数直接接收快照参数，减少共享状态，但当前写法能保持和原来的 `imu_data_solving()` 兼容。

### 9.3 判断所有 IMU 是否就绪

```python
now = time.time()
all_seen = True
for i in range(self.imu_num):
    ts = self.last_seen[i]
    if ts <= 0.0 or (now - ts) > self.seen_timeout:
        all_seen = False
        break
```

`seen_timeout` 当前是 `0.1` 秒。只有所有 IMU 都在最近 0.1 秒内成功解析过数据，`all_seen` 才为真。

这主要服务于首次标定：防止某些 IMU 还没有合法首帧时就开始标定。

### 9.4 标定调度

如果还没有标定完成：

- `all_seen` 为假且还没开始标定：继续等待。
- 一旦 `all_seen` 为真：开始标定。
- 标定开始后，即使中途个别 IMU 短暂丢失，也不打断标定流程。
- 标定计数达到 `calibration_cycles` 后，置 `calibration_done = True`。

这套逻辑解决的是“所有 IMU 真正产生有效数据之后再开始”的问题。

### 9.5 正常工作阶段

标定完成后，后台线程每拿到一个快照就：

```python
self.imu_data_solving()
self.publisher.send_multipart([self.data.qpos.tobytes()])
```

也就是：

1. 调用解算函数更新 `self.data.qpos` 和 `self.data.qvel`。
2. 发布 `qpos` 给外部订阅端。

本文不展开 `imu_data_solving()` 内部计算，只强调它是当前流水线中“IMU 最新数据 -> 关节状态”的转换节点。

## 10. 为什么并行版周期更低

### 10.1 轮询版的问题

`process_pi.py` 每次 `read_imu_data()` 都遍历所有 IMU：

```text
检查 IMU0
检查 IMU1
检查 IMU2
...
检查 IMU6
```

即使某个 IMU 没有新数据，也要执行 `in_waiting` 检查。IMU 数量越多，空检查越多。

另外，轮询版主循环中还直接执行：

```text
读取 -> 标定/解算 -> 发布
```

I/O 和计算在同一个循环里，单轮耗时接近两者相加。

### 10.2 epoll 版的变化

`epoll` 版由内核告诉程序哪些 fd 真的有事件：

```text
epoll 返回：[rfcomm2 可读, rfcomm5 可读]
程序只处理 IMU2 和 IMU5
```

没有事件的串口不会被逐个读取。这样减少了无效系统调用。

### 10.3 解算线程的变化

主线程不再等待 `imu_data_solving()` 完成，而是投递快照后继续下一轮 I/O。系统变成流水线：

```text
第 N 帧：后台线程正在解算
第 N+1 帧：主线程同时继续读 IMU
```

因此吞吐更接近：

```text
max(I/O耗时, 解算耗时)
```

而不是：

```text
I/O耗时 + 解算耗时
```

单帧端到端延迟仍然包含 I/O、排队、解算和发布，但主 I/O 周期会明显降低。

## 11. 当前实现值得注意的点

### 11.1 `epoll.poll(0)` 与 CPU 占用

当前 `epoll.poll(0)` 是最低延迟倾向。如果树莓派 CPU 占用过高，可以考虑：

```python
events = self.epoll.poll(0.001)
```

或：

```python
events = self.epoll.poll(0.002)
```

这样会降低空转，但会引入 1 到 2 ms 量级的最坏等待时间。

### 11.2 `deepcopy` 有额外开销

当前每轮都：

```python
copy.deepcopy(self.imu_data_dict)
```

7 个 IMU 的数据量不大，通常可以接受。但如果后续频率更高，或数据结构变大，可以改成更轻的快照结构，例如固定长度 `numpy.ndarray`，或者只复制 `[quat, gyro]` 的数值数组。

### 11.3 共享状态缺少显式锁

当前主线程会更新：

```python
self.imu_data_dict
self.last_seen
```

后台线程也会读取这些数据。Python 中简单赋值通常不会把对象写坏，但严格实时和安全控制场景建议改成“不可变快照 + 队列传递”，减少跨线程共享可变状态。

### 11.4 ZMQ 只适合发布结果，不建议作为内部实时控制主通道

ZMQ 很适合给可视化、记录、上位机订阅 `qpos`。但如果后续要驱动电机，内部控制链路建议直接用进程内队列或共享内存，避免不必要的序列化和 socket 开销。



# 第二部分：process_pi.py 轮询读取与蓝牙虚拟串口机制

## 1. process_pi.py 的整体轮询流程

[process_pi.py](../Lpms_exo/process_pi.py) 是原始的单线程轮询版本。它和并行版使用了同样的 IMU 数据格式、帧提取函数、payload 解析函数、标定函数和解算函数；区别主要在“什么时候读串口”和“解算是否和 I/O 分离”。

轮询版主循环可以概括为：

```text
while True:
  -> read_imu.read_imu_data()
       -> 依次检查 rfcomm0
       -> 依次检查 rfcomm1
       -> ...
       -> 依次检查 rfcomm6
  -> 前 300 轮做 imu_calibration()
  -> 300 轮后做 imu_data_solving()
  -> ZMQ 发布 qpos
```

入口代码里先创建 `imu_posture`：

```python
IMU_PORTS = [port0, port1, port2, port3, port4, port5, port6]
read_imu = imu_posture(ports=IMU_PORTS)
```

然后在主循环中调用：

```python
read_imu.read_imu_data()
```

如果 `count < 300`，轮询版直接在同一个主循环里做标定；如果 `count > 300`，就在同一个主循环里做解算并通过 ZMQ 发布。

这意味着轮询版的一轮周期包含：

```text
串口轮询读取
  + 帧提取
  + payload 解析
  + 标定或解算
  + ZMQ 发布
```

所以它的周期时间天然是 I/O 和解算串起来的总时间。并行版后面改进的核心，就是把“读取”和“解算/发布”拆开。

## 2. 蓝牙 IMU 到 Linux 虚拟串口的链路

LPMS-B2 蓝牙 IMU 在 Linux/Raspberry Pi 上通常不是直接表现成普通 USB 串口，而是通过蓝牙 RFCOMM 映射成设备节点：

```text
IMU 蓝牙链路
  -> Linux 蓝牙协议栈
  -> RFCOMM 通道
  -> /dev/rfcomm0, /dev/rfcomm1, ...
  -> pyserial 打开这些虚拟串口
```

每一个 `/dev/rfcommN` 都是一个“虚拟串口设备文件”。对 Python 来说，它和普通串口类似，可以用：

```python
serial.Serial('/dev/rfcomm0', 115200, timeout=0.01)
```

打开后，代码通过 pyserial 读取字节流。这里要注意几个概念：

- 蓝牙连接成功不等于应用层已经收到完整 IMU 帧。
- `/dev/rfcommN` 存在不等于该 IMU 的数据已经可解析。
- 呼吸灯或连接状态只能说明链路大体可用，不能说明当前缓存里已经有完整 payload。
- 每个 IMU 应该对应独立的 `/dev/rfcommN`，否则多个设备的数据会进入同一字节流，代码就需要额外的设备 ID 或协议区分。

当前 `process_pi.py` 使用：

```python
port0 = '/dev/rfcomm0'
port1 = '/dev/rfcomm1'
...
port6 = '/dev/rfcomm6'
```

这种“一设备一虚拟串口”的结构，让软件层面可以把每个 IMU 的字节流完全分开处理。

## 3. 串口打开与对象列表

构造函数中，`process_pi.py` 遍历所有端口：

```python
for port in ports:
    ser = serial.Serial(port, baudrate, timeout=0.01)
    ser.reset_input_buffer()
    self.ser_list.append(ser)
    self.buf_list.append(b"")
```

这里建立了两个一一对应的列表：

```text
ser_list[0] -> /dev/rfcomm0 的 Serial 对象
buf_list[0] -> /dev/rfcomm0 的残留字节缓存

ser_list[1] -> /dev/rfcomm1 的 Serial 对象
buf_list[1] -> /dev/rfcomm1 的残留字节缓存
```

`ser.reset_input_buffer()` 用来清空打开串口时内核/驱动里可能残留的旧字节。这样做可以减少“上一次连接残留半帧”影响当前解析的概率。

不过它只能清掉打开那一刻已经在输入缓冲里的字节，不能保证后续蓝牙链路不会出现断帧、半帧、黏包。所以后面仍然必须有 `buf_list` 和 `extract_frames()`。

## 4. read_imu_data 的顺序轮询方式

轮询版 `read_imu_data()` 的核心是一个 `for` 循环：

```python
for imu_idx in range(self.imu_num):
    ser = self.ser_list[imu_idx]
    buf = self.buf_list[imu_idx]
    n = ser.in_waiting
    if n > 0:
        chunk = ser.read(ser.in_waiting)
        buf += chunk
        frames, buf = self.extract_frames(buf)
        self.buf_list[imu_idx] = buf
```

这段代码每调用一次，就会按照 IMU 编号从 0 到 6 顺序检查所有串口。它不是线程并行，也不是事件驱动，而是用户态主动问每个串口：

```text
你现在有多少字节可读？
```

如果 `in_waiting == 0`，说明当前这个串口没有新字节，本轮就跳过它。如果 `in_waiting > 0`，就把已有字节读出来，追加到该 IMU 的缓存。

这种方式的好处是结构简单、容易调试、没有线程同步问题；缺点是每一轮都要检查所有串口，即使大多数串口没有数据，也要做一次检查。IMU 数量越多，空检查越多，周期抖动也更容易被解算、ZMQ 发布和串口状态影响。

## 5. 为什么轮询不会让多个 IMU 数据“打架”

多个 IMU 不会在代码里互相混流，原因有两层。

第一层是 Linux 设备层隔离：

```text
IMU0 -> /dev/rfcomm0 -> ser_list[0]
IMU1 -> /dev/rfcomm1 -> ser_list[1]
IMU2 -> /dev/rfcomm2 -> ser_list[2]
```

每个 `/dev/rfcommN` 都是独立设备文件，有自己的内核输入缓冲。只要系统绑定关系正确，`ser_list[0].read()` 就只读 `/dev/rfcomm0` 的字节，不会读到 `/dev/rfcomm1` 的字节。

第二层是应用层缓存隔离：

```text
/dev/rfcomm0 的残留半帧只放进 buf_list[0]
/dev/rfcomm1 的残留半帧只放进 buf_list[1]
```

`extract_frames()` 每次只处理一个 IMU 的缓存。即便某个 IMU 本轮只到了一半数据，也只是留在自己的 `buf_list[imu_idx]` 中，等待下一次该串口有数据时继续拼接。

所以“轮询”只是读取顺序上的轮流检查，不是把多个串口的数据合并到一个缓冲区里轮流解析。

## 6. 缓存机制：为什么必须有 buf_list

串口和蓝牙传输的是连续字节流，不是 Python 层面的“帧对象”。一次 `read()` 可能读到：

```text
情况 A：半帧
  [frame 前半段]

情况 B：刚好一帧
  [frame]

情况 C：多帧黏在一起
  [frame1][frame2][frame3]

情况 D：残留字节 + 新帧
  [noise][frame]

情况 E：半帧 + 下一轮补齐
  第一次 read: [frame 前半段]
  第二次 read: [frame 后半段]
```

如果没有 `buf_list`，代码只能解析本次 `read()` 读到的字节，一遇到半帧就会丢数据。`buf_list` 的作用就是把“本轮读到的新字节”和“上轮没解析完的残留字节”拼起来：

```python
buf += chunk
frames, buf = self.extract_frames(buf)
self.buf_list[imu_idx] = buf
```

`frames` 是已经切出的完整帧，新的 `buf` 是还不够完整的一段残留。下一轮再读到字节时，会继续接在这个残留后面。

## 7. 帧提取机制

`process_pi.py` 和 `process_pi_parallel.py` 的 `extract_frames()` 逻辑基本一致。

它按下面顺序工作：

```text
在缓存中找帧头 0x3A
  -> 确认至少有 7 字节 header
  -> 从 header 第 5-6 字节读 payload 长度
  -> 计算完整帧长度
  -> 如果缓存长度不够，保留等待下次
  -> 如果长度够，检查帧尾是否为 0x0D 0x0A
  -> 帧尾正确则加入 frames
  -> 帧尾不正确则从下一个字节重新找帧头
```

关键代码：

```python
start = buf.find(b'\x3A')
data_len = int.from_bytes(buf[start + 5:start + 7], 'little')
frame_len = 7 + data_len + 4
frame = buf[start:start + frame_len]
if frame.endswith(b'\x0D\x0A'):
    frames.append(frame)
    buf = buf[start + frame_len:]
else:
    buf = buf[start + 1:]
```

这里的 `frame_len = 7 + data_len + 4` 表示：

```text
header 7 字节
payload data_len 字节
LRC 2 字节
tail 2 字节
```

帧尾 `0x0D 0x0A` 是用于确认帧边界的最后一道检查。帧尾不对时，代码会向后移动一个字节继续搜索，避免因为一段错位数据导致后面所有帧都解析失败。

## 8. payload 解析机制

切出完整帧后，轮询版只解析最新一帧：

```python
latest_frame = frames[-1]
data_len = int.from_bytes(latest_frame[5:7], 'little')
payload = latest_frame[7:7 + data_len]
parsed_data = self.parse_lpms_payload(payload)
```

`parse_lpms_payload()` 的规则是：

```text
payload[0:2] -> timestamp，uint16 小端，除以 100 得到秒
payload[4:]  -> 按 4 字节小端 float32 解析
floats[0:3] -> gyro_x, gyro_y, gyro_z
floats[3:7] -> quat_w, quat_x, quat_y, quat_z
```

之后代码把 LPMS 输出的 `(w, x, y, z)` 转为 scipy/MuJoCo 前面流程使用的 `(x, y, z, w)`：

```python
q = parsed_data["quat"]
q = [q[1], q[2], q[3], q[0]]
g = list(parsed_data["gyro"])
self.imu_data_dict[imu_idx] = [q, g]
```

只取 `frames[-1]` 的设计是实时优先：如果某次读取中堆积了多帧，旧帧没有必要再逐帧解算，直接用最新姿态能降低延迟。

## 9. 轮询版的重连机制

如果读取过程中发生：

```python
OSError
serial.SerialException
```

轮询版会调用：

```python
self._reopen(imu_idx)
```

重连逻辑是：

```text
关闭旧 serial 对象
等待 delay 秒
重新打开同一个 /dev/rfcommN
reset_input_buffer()
替换 ser_list[imu_idx]
清空 buf_list[imu_idx]
```

轮询版不需要处理 `epoll.unregister()` 或 `epoll.register()`，因为它没有把 fd 注册进 epoll。它每轮都是直接从 `self.ser_list[imu_idx]` 取当前串口对象。

但这也带来一个区别：并行版必须维护 `fd_map` 和 epoll 注册关系；轮询版不用。轮询版结构更简单，但每轮检查成本更固定。

## 10. 轮询版和 epoll 并行版的关系

可以把 `process_pi.py` 理解为最直接的基线版本：

```text
简单可靠
  -> 顺序检查每个串口
  -> 同一线程完成读取、标定、解算、发布
```

`process_pi_parallel.py` 是在这个基础上的低延迟版本：

```text
更低延迟
  -> epoll 只处理就绪串口
  -> 主线程只做 I/O
  -> 解算线程处理标定、解算、发布
  -> 队列只保留最新快照
```

两者共享的底层思想是一样的：

- 每个 IMU 一个虚拟串口。
- 每个串口一个独立缓存。
- 用帧头、长度、帧尾恢复完整帧。
- payload 转成 `quat + gyro`。
- `imu_data_dict` 保存每个 IMU 的最新状态。

不同点在于调度方式：

```text
process_pi.py:
  -> 应用层主动轮询所有串口
  -> I/O 和解算串行

process_pi_parallel.py:
  -> 内核 epoll 通知就绪串口
  -> I/O 和解算通过队列分离
```

因此，理解 `process_pi.py` 的轮询、虚拟串口和缓存机制，是理解并行版为什么能优化的基础。



# 第三部分：神经网络推理与宇树 485 电机驱动工作流

## 1. 后续接入神经网络和电机的推荐架构

后续目标是：

```text
IMU 原始姿态
  -> 关节解算
  -> 神经网络推理
  -> 控制量后处理
  -> 电机驱动
  -> 电机反馈与安全监控
```

推荐不要把神经网络和电机发送直接塞进当前 I/O 主线程。主线程应该继续保持短小，只负责 IMU 的 epoll 读取。更好的结构是增加两个阶段：

```text
IMU epoll 线程
  -> raw_imu_queue(maxsize=1)

解算线程
  -> joint_queue(maxsize=1)

神经网络/控制线程
  -> command_queue(maxsize=1)

电机 I/O 线程
  -> motor_feedback_queue(maxsize=1)
```

每个队列都保留最新项，避免过期数据堆积。

## 2. 推荐线程职责

### 2.1 Thread A：IMU epoll 线程

职责：

- 监听 `/dev/rfcomm0` 到 `/dev/rfcomm6`。
- 读取字节。
- 提取完整帧。
- 解析 `quat` 和 `gyro`。
- 生成 `RawImuSnapshot`。
- 投递到 `raw_imu_queue`。

它不做：

- 关节解算。
- 神经网络推理。
- 电机命令计算。
- 电机发送。

这样才能保证 IMU 读取路径持续很短。

建议快照字段：

```python
RawImuSnapshot:
    seq: int
    host_time: float
    imu_time: list[float]
    quat: array shape (7, 4)
    gyro: array shape (7, 3)
    valid_mask: array shape (7,)
```

`valid_mask` 可以表示每个 IMU 在当前时间窗内是否新鲜，方便后续做降级处理或安全停机。

### 2.2 Thread B：关节解算线程

职责：

- 从 `raw_imu_queue` 取最新 IMU 快照。
- 完成标定状态机。
- 调用现有 `imu_data_solving()` 或重构后的解算函数。
- 从 `self.data.qpos` 和 `self.data.qvel` 中取网络需要的关节状态。
- 生成 `JointSnapshot`。
- 投递到 `joint_queue`。
- 可选：继续通过 ZMQ 发布给可视化端。

建议快照字段：

```python
JointSnapshot:
    seq: int
    host_time: float
    source_imu_seq: int
    qpos: array
    qvel: array
    policy_joints: array
    policy_joint_vel: array
    data_age_ms: float
    calibrated: bool
    valid: bool
```

这里最重要的是 `policy_joints` 的顺序必须和神经网络训练时完全一致。不要直接把全部 `qpos` 随便喂给网络，应该明确一个稳定的观测向量定义。

### 2.3 Thread C：`.pt` 神经网络推理与高层控制线程

职责：

- 从 `joint_queue` 取最新关节状态。
- 维护神经网络需要的历史窗口或滤波状态。
- 做归一化、坐标变换、关节顺序整理。
- 调用已有神经网络。
- 将网络输出转换成电机侧目标，例如目标位置、目标速度、力矩、阻抗参数等。
- 做限幅、斜率限制、安全裁剪。
- 生成 `MotorCommand`。
- 投递到 `command_queue`。

建议快照字段：

```python
PolicyOutput:
    seq: int
    host_time: float
    source_joint_seq: int
    raw_action: array
    clipped_action: array
    confidence: float | None

MotorCommand:
    seq: int
    host_time: float
    source_policy_seq: int
    motor_id: array
    mode: array
    target_pos: array | None
    target_vel: array | None
    target_torque: array | None
    kp: array | None
    kd: array | None
    enable: bool
```

你的神经网络模型是 `.pt` 文件，所以这一层优先按 PyTorch 推理线程来设计：线程启动时加载模型、设置 `eval()`、做一次 warmup；运行时只从 `joint_queue` 取最新关节快照，构造观测，执行 `torch.inference_mode()` 推理，再把动作变成电机命令。PyTorch 的核心张量计算通常会进入 C/C++ 后端，很多操作会释放 GIL，因此先用线程是合理的；如果实测推理线程明显拖慢解算或电机发送，再考虑单独进程、TorchScript、ONNX Runtime 或更底层部署。

### 2.4 Thread D：宇树 485 电机 SDK 线程

职责：

- 以固定频率发送最新电机命令。
- 接收电机反馈。
- 检查通信超时。
- 执行急停和安全降级。
- 生成 `MotorFeedback`。

建议快照字段：

```python
MotorFeedback:
    seq: int
    host_time: float
    motor_id: array
    pos: array
    vel: array
    torque: array
    temperature: array | None
    error_code: array | None
    online_mask: array
```

宇树 485 电机 SDK 建议由一个电机线程独占调用，不要让多个线程同时调用同一个 SDK 对象或同一条 485 总线。电机线程不应该等待神经网络每次都产出新命令，它应该维护“最近一条安全命令”，按固定频率重复发送或做底层闭环。

## 3. `.pt` 模型推理接入工作流

这一节专门针对 `.pt` 模型。它的核心原则是：模型加载和推理放在 `policy_thread`，观测构造要严格复现训练环境，网络输出不能直接驱动电机，必须经过单位转换、限幅、滤波和安全层。

### 3.1 `.pt` 文件的两种常见形式

`.pt` 文件可能有两种保存方式，对应加载方法不同。

第一种是保存了完整模型或 TorchScript 模型：

```python
policy = torch.load(policy_path, map_location=device)
policy.eval()
```

如果是 TorchScript：

```python
policy = torch.jit.load(policy_path, map_location=device)
policy.eval()
```

第二种是只保存了 `state_dict`：

```python
policy = ActorNetwork(...)
state = torch.load(policy_path, map_location=device)
policy.load_state_dict(state)
policy.eval()
```

如果 `.pt` 是 `state_dict`，代码里必须能拿到训练时的网络结构类，例如 `ActorNetwork`。如果 `.pt` 是完整模型或 TorchScript，部署文件可以少一些，但仍然要确认输入维度、输出维度、归一化参数和动作缩放方式。

### 3.2 观测向量必须和训练一致

仓库里已有 [sensor_to_obs_bluetooth.py](../Lpms_exo/sensor_to_obs_bluetooth.py)，其中 `imu_to_obs(pos, vel)` 会把 `qpos/qvel` 转成：

```text
joint_pos
joint_vel
root_pos_w
root_vel_w
```

[test.py](../Lpms_exo/test.py) 里曾经按下面方式拼接 65 维输入：

```python
joint_pos, joint_vel, root_pos_w, root_vel_w = imu_to_obs(qpos, qvel)
obs = np.concatenate([joint_pos, joint_vel, root_pos_w, root_vel_w])
```

如果你的 `.pt` 模型就是按这套观测训练的，后续 `PolicyRunner` 应该复用这套顺序。不要因为当前 `self.data.qpos` 里有更多 MuJoCo 状态，就直接把整个 `qpos` 喂给网络；网络只认识训练时的观测定义。

观测构造要确认这些内容：

- 关节顺序：右髋、左髋、右膝、左膝等顺序必须和训练一致。
- 单位：角度用 rad 还是 degree，要和训练一致。
- 坐标系：`sensor_to_obs_bluetooth.py` 里有 `(x, -z, y)` 这样的坐标变换，不能随意改。
- 根部姿态：当前用 tangent + normal 的 6 维表示，也要和训练一致。
- 归一化：如果训练时有 mean/std 或 normalizer，部署时必须加载同一套参数。
- 历史帧：如果模型训练时用了历史窗口，policy 线程要维护 ring buffer。

### 3.3 policy 线程推理步骤

推荐 `policy_thread` 启动后先完成模型初始化：

```python
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
policy = load_policy(policy_path, device)
policy.eval()

dummy_obs = torch.zeros(obs_dim, dtype=torch.float32, device=device)
with torch.inference_mode():
    _ = policy(dummy_obs.unsqueeze(0))
```

运行时循环：

```python
def _policy_loop(self):
    while True:
        joint = self.joint_queue.get()
        if not joint.valid or not joint.calibrated:
            self._put_latest(self.command_queue, make_safe_command())
            continue

        obs_np = build_observation(joint)
        obs_np = normalize_observation(obs_np)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            action = self.policy(obs.unsqueeze(0))

        action_np = action.squeeze(0).detach().cpu().numpy()
        command = action_to_motor_command(action_np, joint)
        command = apply_command_safety(command)
        self._put_latest(self.command_queue, command)
```

这里建议使用 `torch.as_tensor()`，避免每次都做不必要的数据复制。输出 `action_np` 后立刻脱离 PyTorch，后续限幅和电机命令转换可以用 numpy 或普通 Python 数据结构完成。

### 3.4 网络输出到电机命令

网络输出是什么，要看训练任务定义。常见情况有：

- 输出目标关节位置：需要转换成电机目标位置，再由电机侧 PD 或 SDK 控制。
- 输出目标关节速度：需要转换成电机目标速度，并加速度/速度限幅。
- 输出关节力矩：需要转换成电机力矩或电流命令，并严格限幅。
- 输出归一化动作：需要按训练时的 action scale 和 offset 还原。

对外骨骼而言，最危险的是把网络原始输出直接当电机力矩发出去。推荐固定经过这几层：

```text
raw_action
  -> action scale/offset 还原
  -> 关节侧单位转换
  -> 关节到电机映射
  -> 电机方向符号修正
  -> 幅值限幅
  -> 变化率限制
  -> 低通滤波
  -> 安全状态机裁决
  -> MotorCommand
```

### 3.5 policy 线程是否需要单独进程

先用线程。理由是：

- `.pt` 推理主要由 PyTorch 后端执行，不一定长期占住 GIL。
- 线程之间传递 `JointSnapshot` 和 `MotorCommand` 成本低。
- 调试简单，和当前 `process_pi_parallel.py` 的线程结构一致。

当出现下面情况时，再考虑把 policy 放到进程：

- 推理耗时明显高于目标周期，比如目标 100 Hz，但单次推理经常超过 10 ms。
- 推理时 IMU I/O 或电机发送出现明显抖动。
- 模型很大，CPU 多核利用不足。
- 需要把模型崩溃和电机安全线程隔离。

如果改成进程，建议只把 `PolicyRunner` 独立出去，电机安全线程仍保留在主控制进程或更高优先级进程中。

## 4. epoll 在后续控制里怎么用

`epoll` 适合处理“文件描述符 I/O”，不适合直接处理计算。

适合放进 `epoll` 的对象包括：

- 裸串口 fd。
- SocketCAN 的 CAN socket fd。
- TCP/UDP socket。
- `timerfd` 周期定时器。
- `signalfd` 退出信号。

不适合用 `epoll` 的对象包括：

- 神经网络推理本身。
- numpy 矩阵运算本身。
- 纯 Python 控制算法本身。
- 不暴露 fd 的厂商 SDK 调用。

对当前系统来说，IMU 读取已经非常适合 `epoll`；宇树 485 电机如果是通过官方 SDK 调用，优先看 SDK 是否暴露底层串口 fd：

- 如果 SDK 暴露 fd，并且允许外部做非阻塞读写，可以在电机线程内部使用 `epoll + timerfd`。
- 如果 SDK 只提供 `send()`、`recv()`、`query()` 这类 API，不要强行绕过 SDK 做 epoll，直接让电机线程独占 SDK，并用固定周期循环调用它。

如果可以使用 fd，电机线程可以这样组织：

```text
motor_epoll 注册：
  -> CAN socket fd 或电机串口 fd：接收反馈
  -> timerfd：固定周期发送最新命令
  -> signalfd/eventfd：退出或急停事件
```

这样电机线程可以同时做到：

- 有反馈立刻读。
- 到固定控制周期立刻发命令。
- 收到急停事件立刻切断输出。

如果不能使用 fd，也可以保留“固定周期”思想：

```text
motor_thread:
  -> 每 1 ms 或 2 ms 醒来
  -> 取 command_queue 最新命令
  -> 检查命令是否过期
  -> 调用宇树 485 SDK 发送
  -> 调用宇树 485 SDK 读取或查询反馈
  -> 更新 motor_feedback_queue
```

## 5. 推荐控制周期设计

实际频率要根据 IMU、神经网络和电机总线能力确定。一个常见设计是：

```text
IMU 读取：100 到 200 Hz，事件驱动，谁有数据读谁
关节解算：跟随 IMU 最新快照，通常 100 到 200 Hz
神经网络：50 到 200 Hz，取决于模型大小
电机发送：500 到 1000 Hz，固定周期
电机反馈：有反馈就读，或跟随驱动器反馈周期
```

如果电机控制频率高于神经网络频率，不需要让神经网络跑到 1000 Hz。可以这样做：

```text
神经网络每 10 ms 更新一次目标动作
电机线程每 1 ms 发送一次命令
电机线程在两次网络输出之间保持上一条目标，或做插值/低层 PD
```

这比“网络每次推理后才发一次电机命令”更稳定。

## 6. 最小改造方案

如果想在现有 `process_pi_parallel.py` 上小步接入，可以按以下顺序改。

### 6.1 在构造函数中增加两个队列和线程

```python
self.joint_queue = queue.Queue(maxsize=1)
self.command_queue = queue.Queue(maxsize=1)
self.motor_feedback_queue = queue.Queue(maxsize=1)

self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
self.policy = load_pt_policy(policy_path, self.device)
self.policy.eval()

self.policy_thread = threading.Thread(target=self._policy_loop, daemon=True)
self.motor_thread = threading.Thread(target=self._motor_loop, daemon=True)
self.policy_thread.start()
self.motor_thread.start()
```

### 6.2 在 `_solver_loop()` 解算后投递关节快照

当前代码：

```python
self.imu_data_solving()
self.publisher.send_multipart([self.data.qpos.tobytes()])
```

可以扩展为：

```python
self.imu_data_solving()
joint_snapshot = self._build_joint_snapshot()
self._put_latest(self.joint_queue, joint_snapshot)
self.publisher.send_multipart([self.data.qpos.tobytes()])
```

这里 `_build_joint_snapshot()` 应该从 `self.data.qpos` 和 `self.data.qvel` 中取出网络训练时定义好的观测量。

### 6.3 增加 `_policy_loop()`

伪代码：

```python
def _policy_loop(self):
    while True:
        joint = self.joint_queue.get()
        if not joint.valid or not joint.calibrated:
            self._put_latest(self.command_queue, make_safe_command())
            continue

        obs = build_observation(joint)
        obs = normalize(obs)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            action = self.policy(obs.unsqueeze(0))

        action = action.squeeze(0).detach().cpu().numpy()
        command = action_to_motor_command(action)
        command = apply_limits(command)
        self._put_latest(self.command_queue, command)
```

注意点：

- `build_observation()` 的输入顺序、单位、归一化参数必须和训练一致。
- 输出必须限幅，不能把网络原始输出直接发给电机。
- 如果关节数据过期，应该不推理或输出安全命令。

### 6.4 增加 `_motor_loop()`

伪代码：

```python
def _motor_loop(self):
    latest_command = make_safe_zero_command()
    next_tick = time.perf_counter()

    while True:
        command = try_get_latest(self.command_queue)
        if command is not None:
            latest_command = command

        if command_is_stale(latest_command):
            latest_command = make_safe_zero_command()

        sdk_command = build_unitree_485_command(latest_command)
        self.unitree_motor.send(sdk_command)
        feedback = self.unitree_motor.recv_or_query()
        self._put_latest(self.motor_feedback_queue, feedback)

        sleep_until_next_tick()
```

这里的 `send()`、`recv_or_query()` 只是占位名，实际实现要替换成宇树 485 SDK 的真实 API。关键点是：SDK 调用集中在 `_motor_loop()` 一个线程里，上层线程不直接碰 SDK。

## 7. 更推荐的中期重构方案

最小改造能跑起来，但随着神经网络和电机加入，建议逐渐拆成几个类：

```text
ImuEpollReader
  -> 只负责 rfcomm epoll、切帧、payload 解析

PostureSolver
  -> 只负责标定和 IMU 到关节状态

PolicyRunner
  -> 只负责观测构造、归一化、模型推理、动作后处理

MotorController
  -> 只负责电机通信、固定周期发送、反馈解析、安全状态机

RealtimePipeline
  -> 负责启动/停止线程、连接队列、日志和异常处理
```

这样每一层的输入输出清晰，后续调试也更容易定位：

- IMU 问题看 `ImuEpollReader`。
- 姿态问题看 `PostureSolver`。
- 网络输出问题看 `PolicyRunner`。
- 电机问题看 `MotorController`。

## 8. 安全工作流

接入电机后，实时系统必须有安全状态机。建议至少包含这些条件。

### 8.1 启动阶段

```text
打开 IMU 串口
打开电机通信
确认电机在线
确认急停未触发
等待所有 IMU 有效首帧
完成标定
神经网络 warmup
电机进入使能前安全模式
人工确认或软件条件满足后 enable
```

### 8.2 运行阶段

每一轮都检查：

- IMU 数据是否过期。
- 解算结果是否有效。
- 神经网络输出是否有限且未超限。
- 电机反馈是否在线。
- 电机温度、电压、电流、错误码是否正常。
- 命令变化率是否过大。
- 急停信号是否触发。

任何条件不满足，都应该进入安全输出：

```text
位置模式：保持当前位置或回到安全位姿
速度模式：目标速度归零
力矩模式：目标力矩归零
阻抗模式：降低刚度或关闭使能
```

具体策略取决于外骨骼硬件，但不能让网络原始输出绕过安全层。

### 8.3 超时建议

可以先设置保守阈值：

```text
IMU 快照超过 50 ms 未更新：暂停推理或输出安全命令
关节快照超过 50 ms 未更新：输出安全命令
网络命令超过 100 ms 未更新：电机线程不再沿用旧命令
电机反馈超过 20 ms 未更新：进入通信异常状态
```

这些数字需要结合实际频率调整，但原则是：任何关键数据过期，都不能继续盲目驱动。

## 9. 数据流中的时间戳和序号

后续接神经网络和电机时，建议每一层快照都带：

```text
seq：递增序号
host_time：本机 time.perf_counter() 或 time.time()
source_seq：来自上一层的数据序号
data_age_ms：当前使用的数据年龄
valid：该快照是否可用于控制
```

这样日志里可以追踪一条命令来自哪一帧 IMU：

```text
IMU seq 10023
  -> Joint seq 9981
  -> Policy seq 9950
  -> MotorCommand seq 9950
```

调延迟时这比只看平均周期更有用。

## 10. 推荐的内部队列工具函数

为了保持“只保留最新数据”，可以统一封装：

```python
def put_latest(q, item):
    try:
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put_nowait(item)
    except queue.Full:
        pass
```

所有实时队列都使用这个模式：

```text
raw_imu_queue
joint_queue
command_queue
motor_feedback_queue
```

日志队列可以例外。日志通常需要完整记录，不一定只保留最新，但日志线程优先级应低于控制线程，不能反过来拖慢控制。

## 11. 线程和进程的区别

这部分很关键，因为后续系统里同时有 IMU I/O、解算、`.pt` 推理和电机 SDK。线程和进程都能“并行组织代码”，但它们解决的问题不一样。

线程是在同一个 Python 进程里跑的多个执行流：

```text
同一个进程
  -> Thread A: IMU epoll
  -> Thread B: solver
  -> Thread C: policy
  -> Thread D: motor SDK
```

这些线程共享同一个内存空间，所以传递数据很方便，一个 `queue.Queue(maxsize=1)` 就能把最新快照交给下一层。缺点是 Python 有 GIL：同一时刻通常只有一个线程在执行 Python 字节码。好消息是，I/O 等待、numpy、PyTorch 等很多底层 C/C++ 计算会释放 GIL，所以这个项目里“线程先行”是合理的。

进程是多个独立 Python 解释器：

```text
主进程
  -> IMU + solver + motor safety

policy 进程
  -> 加载 .pt 模型并推理
```

进程之间内存不共享，必须通过 `multiprocessing.Queue`、Pipe、共享内存或 socket 传数据。优点是能绕开 GIL、隔离崩溃、利用多核更彻底；缺点是通信成本更高，数据结构要序列化或放共享内存，启动、退出和调试都更复杂。

### 11.1 适合线程

- 串口、CAN、socket 等 I/O。
- PyTorch、numpy 等底层释放 GIL 的计算。
- 轻量控制逻辑。
- 需要共享状态、快速传递小数据的实时流水线。
- ZMQ 发布和日志。

对当前项目，下面这些先放线程里比较自然：

```text
IMU epoll 读取线程
关节解算线程
.pt policy 推理线程
宇树 485 SDK 电机线程
日志/可视化线程
```

### 11.2 适合进程

- 纯 Python 的重计算。
- `.pt` 神经网络推理明显阻塞其他线程。
- 希望充分利用多核，并且能接受进程间通信成本。
- 希望把电机安全控制和上层算法隔离。
- 某个第三方库不稳定，崩溃后不希望拖垮主控制进程。

但电机 SDK 是否放进独立进程要谨慎。电机安全输出应该尽量靠近硬件，如果把电机通信放到另一个进程，主进程和电机进程之间一旦通信卡住，安全逻辑会变复杂。更稳妥的顺序是：

```text
先：电机 SDK 在线程中独占运行
再：测量 SDK 调用是否阻塞过久
最后：确实需要隔离时，再把 MotorController 独立成进程
```

### 11.3 对当前项目的建议

先用线程完成闭环，结构清晰后再看瓶颈：

```text
第一步：epoll IMU + solver thread + policy thread + motor thread
第二步：测量每层耗时和数据年龄
第三步：如果 .pt policy 或 solver 成为瓶颈，再考虑进程或 TorchScript/ONNX Runtime
```

不要过早把所有东西都改成多进程，否则调试成本会明显上升。

## 12. 宇树 485 电机 SDK 的接入方式

### 12.1 总体原则

你的电机使用的是宇树 485 通信的电机 SDK，因此推荐把电机通信封装成一个独立的 `MotorController` 或 `_motor_loop()` 线程。这个线程独占 SDK 对象和 485 串口/总线，上层只通过 `command_queue` 给它最新目标命令，它再通过 `motor_feedback_queue` 把电机反馈发回系统。

不要让 policy 线程直接调用宇树 SDK，也不要让解算线程直接发送电机命令。原因有四个：

- 485 总线通常是半双工或严格时序通信，多线程同时调用 SDK 容易造成请求/反馈错位。
- SDK 调用可能阻塞，不能拖慢 IMU 读取或关节解算。
- 电机发送需要固定周期，而神经网络推理周期可能有抖动。
- 电机安全层应该独立存在，能够在网络输出超时后继续发送安全命令。

所以电机侧的基本边界是：

```text
PolicyRunner
  -> 只产生 MotorCommand

MotorController
  -> 独占宇树 485 SDK
  -> 固定周期发送最新安全命令
  -> 读取反馈
  -> 执行超时和急停逻辑
```

### 12.2 推荐电机线程工作流

电机线程初始化：

```text
打开宇树 485 SDK/串口
扫描或配置电机 ID
读取一次电机状态，确认在线
设置控制模式或清除错误
发送零力矩/安全命令
进入循环
```

循环中：

```python
def _motor_loop(self):
    latest_command = make_safe_command()
    latest_command_time = time.perf_counter()
    period = 0.001  # 例如 1 ms，根据 SDK 和 485 总线能力调整
    next_tick = time.perf_counter()

    while self.running:
        cmd = try_get_latest(self.command_queue)
        if cmd is not None:
            latest_command = cmd
            latest_command_time = time.perf_counter()

        if time.perf_counter() - latest_command_time > self.command_timeout:
            latest_command = make_safe_command()

        safe_command = apply_motor_safety(latest_command)
        self.unitree_motor.send(safe_command)
        feedback = self.unitree_motor.recv_or_query()
        self._put_latest(self.motor_feedback_queue, feedback)

        next_tick += period
        sleep_time = next_tick - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_tick = time.perf_counter()
```

实际 SDK 的函数名可能不是 `send()` 和 `recv_or_query()`，这里表达的是职责：发送最新命令、读取反馈、更新反馈队列。SDK 如果内部已经包含“发送并返回反馈”的组合函数，也可以每个周期调用一次组合函数。

### 12.3 宇树 485 命令转换

`PolicyRunner` 输出的是网络动作，`MotorController` 需要的是 SDK 能接受的电机命令。中间需要一层映射：

```text
policy action
  -> 关节侧目标：q_des / qd_des / tau_des
  -> 外骨骼关节到电机 ID 映射
  -> 减速比、方向符号、零位偏置转换
  -> 宇树 SDK 命令字段
```

建议把这些配置写成表，而不是散落在代码里：

```python
MOTOR_MAP = {
    "right_knee": {
        "motor_id": 0,
        "sign": 1.0,
        "gear_ratio": 1.0,
        "zero_offset": 0.0,
        "torque_limit": 0.5,
        "velocity_limit": 2.0,
    },
    "left_knee": {
        "motor_id": 1,
        "sign": -1.0,
        "gear_ratio": 1.0,
        "zero_offset": 0.0,
        "torque_limit": 0.5,
        "velocity_limit": 2.0,
    },
}
```

映射时至少做这些保护：

- 目标位置限幅。
- 目标速度限幅。
- 目标力矩限幅。
- 单周期变化率限制。
- 方向符号检查。
- 电机零位偏置检查。
- 命令中不能出现 `nan` 或 `inf`。

### 12.4 什么时候考虑电机进程

默认先用电机线程。如果出现下面情况，再考虑把宇树 485 SDK 放到单独进程：

- SDK 偶发卡死，线程无法可靠恢复。
- SDK 调用会长时间阻塞，影响同进程其它线程。
- SDK 或底层动态库崩溃会直接杀掉 Python 进程。
- 需要给电机控制进程设置更高系统优先级。

如果拆成进程，建议电机进程仍然自己维护安全状态机：一旦主进程的命令超过超时阈值，就自动转入安全命令，而不是等待主进程继续指挥。

## 13. 推荐的最终运行工作流

完整运行时可以设计成以下顺序。

### 13.1 初始化

```text
加载配置
加载神经网络
加载归一化参数
加载关节到电机映射
打开 IMU rfcomm 串口
打开电机通信接口
启动 IMU epoll 线程
启动解算线程
启动 policy 线程
启动 motor 线程
启动日志/可视化线程
```

### 13.2 标定

```text
IMU 线程持续收帧
解算线程等待所有 IMU last_seen 新鲜
开始标定
标定完成后输出 calibrated=True 的 JointSnapshot
policy 线程收到 calibrated=True 后才开始推理
motor 线程在 calibrated=True 和 enable=True 前只发送安全命令
```

### 13.3 闭环运行

```text
IMU epoll 线程：持续更新 RawImuSnapshot
解算线程：持续生成 JointSnapshot
policy 线程：持续生成 MotorCommand
motor 线程：固定周期发送最新 MotorCommand，并持续接收 MotorFeedback
安全状态机：任何超时或错误都会覆盖 MotorCommand 为安全命令
ZMQ/日志线程：异步记录 qpos、qvel、action、motor feedback
```

### 13.4 退出

```text
收到 Ctrl+C 或急停
motor 线程立即发送安全命令
关闭电机使能
关闭 ZMQ
注销 epoll fd
关闭 IMU 串口
关闭电机通信接口
落盘日志
```

## 14. 对当前代码的具体接入建议

短期最务实的接入点是在 `_solver_loop()` 的正常工作阶段：

```python
self.imu_data_solving()
```

之后立刻构造神经网络需要的关节快照：

```text
self.data.qpos / self.data.qvel
  -> extract_policy_observation()
  -> joint_queue
```

不要在 `_solver_loop()` 里直接推理和发送电机命令。原因是：

- 推理耗时可能抖动。
- 电机发送应该有固定周期。
- 安全状态机应该独立于解算线程。
- 解算线程阻塞会反过来让 `solve_queue` 消费变慢。

所以推荐：

```text
_solver_loop()
  -> 只负责产出 JointSnapshot

_policy_loop()
  -> 只负责 JointSnapshot 到 MotorCommand

_motor_loop()
  -> 只负责固定周期发送 MotorCommand
```

这和当前 `solve_queue(maxsize=1)` 的思想完全一致：每层都只拿最新数据，避免为了追旧帧牺牲实时性。

## 15. 一句话总结

当前 `process_pi_parallel.py` 已经把最关键的低延迟骨架搭起来了：主线程用 `epoll` 做 IMU 事件驱动读取，后台线程用 `Queue(maxsize=1)` 接收最新快照并完成标定/解算/发布。后续接神经网络和电机时，不应把所有工作塞回一个循环，而应继续沿用“epoll 管 I/O、线程管计算、队列传最新快照、电机线程固定周期输出、安全层最后兜底”的流水线。
