# 网络切分中的手机：应用需要添加什么

*语言：[English](MOBILE_SPLIT.md) · [Русский](MOBILE_SPLIT.ru.md) · **中文***

状态：ABI 已在 0.5.70 中完成并经过测量。应用端的工作尚未开始。
相关文档：`MULTI_DEVICE_ROADMAP.ru.md`（§3.5，Phase 4b；俄文）、
`CMF_V2_SPEC.zh.md`、头文件 `crates/cortiq-ffi/include/cortiq.h`。

---

## 0. 究竟测了什么——以及为什么角色比切分更重要

测试平台：Xiaomi 12 Lite（SD778G，4×A78 2.4 GHz + 4×A55，7.3 GB 内存，其中约
2 GB 空闲），通过 USB-C 连接，并与 MacBook M4 24 GB 处于同一个 5 GHz
802.11ax Wi-Fi 网络。两个方向都运行真实的 `cortiq worker`，而不是模拟。

**按层切分不会让单个 token 变快。** token 依次逐层计算；分布带来的是内存，
而不是并行。两个方向、两种传输方式都测过：任何两端都参与计算的配置，都比
最快的一端单独运行要慢。Bonsai 1.7B：Mac 28.6 tok/s，手机 14.3（稳态），经
USB 对半切分——14.0。

由此得出**四种角色**，应用应当选择角色，而不是"负载百分比"：

| 角色 | 带来什么 | 实测 |
|---|---|---|
| **瘦客户端**——桌面端承担全部计算，手机只负责显示 | 手机上根本放不下的模型 | 手机仅有 2 GB 空闲内存时运行 34.7B MoE：**16.3 tok/s** |
| **预填充卸载**——由桌面端计算长 prompt | 首 token 时间 | 1800 个位置：86.9 s → **11.3 s（7.7×）** |
| **内存扩展**——手机承担桌面端尾部的若干层 | 比桌面端内存更大的模型 | 只有模型放不下时才有收益；否则纯属负收益 |
| **副本**——手机独立处理整个请求 | 吞吐量 | 即现在的"服务器"界面，它已经是对的 |

传输方式按**尾部延迟**排名，而不是按带宽。4 KB 往返（hidden 为 2048 时恰好
一个 hidden 向量）：

| 路径 | p50 | p90 | p99 |
|---|---|---|---|
| USB | **1.89 ms** | 2.19 | 2.94 |
| Wi-Fi 5 GHz ax，−53 dBm，链路 680 Mbit/s | 8.95 | 11.39 | **94.81** |
| token 之间暂停 100 ms 的 Wi-Fi | 9.56 | **85.14** | 189.8 |

每个 token 一次往返，意味着用户看到的是 p99。线缆在中位数上比优秀的 Wi-Fi
快五倍，在尾部快三十倍。

---

## 1. ABI 中新增的内容（0.5.70）

三个函数，全部是增量添加。不调用它们的应用照旧工作。

```c
int32_t     cortiq_worker_start(const char *config_json);
int32_t     cortiq_set_peer(const char *config_json);
const char *cortiq_peer_stats(void);
```

**`cortiq_worker_start`**——把 `cortiq worker` 变成一次函数调用：手机把自己那份
模型副本中的一段层交给别人的协调器（coordinator）计算。监听器运行在后台
线程上。

```json
{"model": "/data/.../model.cmf", "listen": "0.0.0.0:9911", "token": "secret"}
```

端口绑定和文件检查**在调用者线程上**完成，因此端口被占用或文件缺失会返回
`-1`，并在 `cortiq_last_error()` 中给出可以展示给用户的文本，而不是一个悄无
声息死掉的线程。除 loopback 以外的任何地址都**必须**提供 token：没有它，
worker 拒绝监听。

**`cortiq_set_peer`**——方向相反：此后的每次生成都经由持有**同一**文件的
对端（peer）进行。

```json
{"addr": "192.168.1.5:9911", "token": "secret",
 "split": 0, "dtype": "f16", "head": true}
```

- `split`——对端计算的第一层。`0` = 全部层，此时本端只保留 tokenizer。默认值
  为层栈的一半。
- `head`——对端还拥有 `lm_head` 和采样器，并返回 **token id**。
- `dtype`——`f32` 逐字节复现本地文本，`f16` 把线路流量减半，并且会合理地改变
  文本。

`null` 或 `{}` 会清除设置并回到本地模式。连接在第一次生成时才懒加载建立——
不可达的对端在那时失败，并给出消息。

**`cortiq_peer_stats`**——对端**此刻**的状态，JSON 格式：

```json
{"thermal_mc":34300,"powered":null,"cpu_khz_cur":691200,
 "cpu_khz_max":2400000,"mem_avail_kb":1467000,"threads":4,
 "platform":"android/aarch64"}
```

`null` 表示"平台不提供该数据"，而不是零。不要在那里填零：读到 0 MHz 的调度器
会把一个活着的节点搁置。

---

## 2. `head` 标志——这里最便宜的加速

头部（`final norm` + `lm_head` + 采样器）**不会缩小**，无论移走多少层。在手机
上它占 73 ms 的 token 时间中的 29 ms——也就是说，作为客户端的手机受限于它
自己的头部，与桌面端的速度无关。设置 `head: true` 后，头部移到对端，传回来的
是 token id。

手机作协调器 + Mac 作 worker，所有层都在 Mac 上，经 USB：

| 模型 | `head: false` | `head: true` | |
|---|---|---|---|
| bonsai 1.7B | 12.6–13.0 tok/s | **26.0–27.3** | 2.06× |
| Qwen3.6 35B-A3B q2tp (34.7B) | 9.1–9.4 | **15.7–16.3** | 1.72× |

在这种模式下，每个 token 的线路流量是 **16 字节**（去 12、回 4），而不是
4 KB，`net:` 一行显示 "99% of decode wall"：手机只是等待并把 token 还原成
文本（detokenize）。

为什么不能**只**移走头部而不带采样器：这些模型的 logits 有 151,669 个数，
f16 下每个 token 300 KB。按实测的 Wi-Fi 带宽 14.6 MB/s 计算，每个 token 需要
20 ms——比省下的还多。必须传输的是 id。

**一致性：** 在相同切分、两端相同后端的情况下，`head: true` 给出的文本与
`head: false` 逐字节相同——无论是 greedy 还是固定 seed 的采样（采样器配置在
Assign 中发送到对端；否则对端的默认温度会悄悄改写回答）。注意：此模式把头部
移到**对端的设备**上，因此如果两端后端不同（手机用 CPU，桌面端用 Metal），
文本可能合理地发生变化——这与任何跨设备切分的规则相同。

---

## 3. 应用中要做的事，逐步说明

### 第 1 步。重新构建原生库

各 ABI 的构建矩阵并**不**相同，下面两个特殊之处只要漏掉一个，就会悄无声息地
破坏应用。

```sh
# 64 位 ABI——启用 gpu 特性。它不是默认特性：不带它的构建会得到
# 体积只有一半、没有 Vulkan 的库和应用，而且不报任何错误。
cargo ndk -t arm64-v8a -t x86_64 --platform 26 \
  build --release -p cortiq-ffi --features gpu

# armeabi-v7a——不带 gpu（应用中现有的库就是这样构建的），并且使用
# 16 KB 页：链接器默认使用 0x1000 对齐，这样的库
# 在 Android 15 上无法加载，也无法通过 Play 审核。
RUSTFLAGS="-C link-arg=-Wl,-z,max-page-size=16384" \
  cargo ndk -t armeabi-v7a --platform 26 build --release -p cortiq-ffi

# iOS——同样启用 gpu，且两个目标都要：真机和模拟器（arm64）
cargo build --release -p cortiq-ffi --target aarch64-apple-ios --features gpu
cargo build --release -p cortiq-ffi --target aarch64-apple-ios-sim --features gpu
```

复制**之前**先检查——三个 ABI 都要有这三个符号，并且对齐为 0x4000
（`NDK/toolchains/llvm/prebuilt/*/bin`）：

```sh
llvm-nm -D libcortiq_ffi.so | grep -c "cortiq_\(worker_start\|set_peer\|peer_stats\)"   # 3
llvm-readelf -l libcortiq_ffi.so | awk '/LOAD/{print $NF}' | sort -u        # 0x4000
```

大小也是一种检查：arm64 ≈ 14.1 MB、x86_64 ≈ 14.5 MB（带 gpu），v7a ≈ 4.5 MB
（不带），两个 iOS 库都 ≈ 88.7 MB（带 gpu；不带时为 53.6——如果得到的是这个
大小，说明特性丢了）。

**普通的 `nm` 看不到 iOS 归档中的符号，这并不是故障。** release 配置使用
`lto = "thin"` 构建，因此归档成员是 LLVM bitcode，`nm` 即使对
`_cortiq_version` 也返回空。要通过链接来检查：

```sh
printf 'int32_t cortiq_set_peer(const char*);\nint main(){return cortiq_set_peer("{}");}\n' > probe.c
xcrun -sdk iphoneos clang -target arm64-apple-ios13.0 probe.c libcortiq_ffi.a \
  -framework Metal -framework Foundation -framework CoreFoundation \
  -framework CoreGraphics -framework QuartzCore -framework IOSurface -o probe
```

此列表中的 `CoreGraphics` 是必需的：wgpu 的 Metal surface 依赖
`CGColorSpace*`，没有它，链接会因这些符号而失败，而不是因为我们的符号。

把 `libcortiq_ffi.so` 复制到 `android/app/src/main/jniLibs/<abi>/`，并用
`crates/cortiq-ffi/include/cortiq.h` 更新 `native/cortiq_ffi.h`。

**两端必须使用同一协议版本。** 握手时会比对 `WIRE_VERSION`（当前为 7），不
一致时以 "wire version N ≠ 7 — upgrade one side" 的文本拒绝连接。这是一项
特性：不兼容会立即显现，而不是以输出中的乱码出现。

### 第 2 步。绑定

按照现有 `lib/data/services/inference/native_engine.dart` 的风格：

```dart
typedef _WorkerStartNative = ffi.Int32 Function(ffi.Pointer<Utf8>);
typedef _WorkerStartDart = int Function(ffi.Pointer<Utf8>);
typedef _SetPeerNative = ffi.Int32 Function(ffi.Pointer<Utf8>);
typedef _SetPeerDart = int Function(ffi.Pointer<Utf8>);
typedef _PeerStatsNative = ffi.Pointer<Utf8> Function();

late final _workerStart = _lib
    .lookupFunction<_WorkerStartNative, _WorkerStartDart>('cortiq_worker_start');
late final _setPeer =
    _lib.lookupFunction<_SetPeerNative, _SetPeerDart>('cortiq_set_peer');
late final _peerStats =
    _lib.lookupFunction<_PeerStatsNative, _PeerStatsNative>('cortiq_peer_stats');

/// 让后续的生成经由桌面端进行。`split: 0` + `head: true`
/// 即瘦客户端模式：手机只做分词。
bool useDesktop(String addr, String token,
    {int split = 0, bool head = true, String dtype = 'f16'}) {
  final cfg = jsonEncode({
    'addr': addr, 'token': token, 'split': split,
    'head': head, 'dtype': dtype,
  }).toNativeUtf8();
  try {
    return _setPeer(cfg) == 0;
  } finally {
    calloc.free(cfg);
  }
}

/// 回到本地模型。
void useLocal() => _setPeer(ffi.nullptr);

Map<String, dynamic> peerStats() =>
    jsonDecode(_peerStats().toDartString()) as Map<String, dynamic>;
```

`cortiq_set_peer` 不会阻塞，也不访问网络——连接在第一次生成时建立。错误从
目前显示生成错误的同一位置显示：通过 `cortiq_last_error()`。

### 第 3 步。"伴侣"界面

已经有用的最小功能集：

1. 地址和 token 输入框（或二维码——"服务器"界面已经会绘制二维码，这里需要的
   是反向流程：桌面端显示，手机扫描）。
2. 角色切换：**"在本机计算"** / **"在桌面端计算"**。
3. 状态行，诚实而切中要点——什么在哪里计算、经由哪条线路。
4. "检查"按钮——发一个简短请求；如果对端没有启动，显示错误文本，而不是
   转圈。

模型要求：**两端必须放着同一个 `.cmf`**。握手时按 `dir_hash` 校验身份，外来
文件会被拒绝——永远不会拼出一个"嵌合体"。对 UI 而言，这意味着应当把桌面端的
模型列表与本地列表比对，而不是提供"连接任意模型"的选项。

即使 `split: 0`，手机上也需要这个文件——tokenizer 和聊天模板要从中读取。它在
磁盘上占用完整的模型大小，在内存中只占被访问到的页面（34.7B q2tp：磁盘上
12.9 GB，在 2 GB 空闲内存下运行）。

### 第 4 步。worker 角色——仅限前台

`cortiq_worker_start` 会启动监听器，但手机必须保持运行、不进入休眠（doze）：
前台服务 + wake lock。应用中已有 `core/util/foreground_task.dart` 和
`core/util/keep_awake.dart`——与"服务器"模式走同一条路径。

### 第 5 步。轮询状态

在对话轮次之间（而不是在解码循环中）调用 `cortiq_peer_stats()`，并显示
`cpu_khz_cur / cpu_khz_max`。如果活跃解码时低于约 50%，说明对端的调频器
（governor）没有唤醒——这可以修复，见陷阱 3。

---

## 4. 陷阱——每一个都是测量换来的

**1. Wi-Fi 只能靠批处理解决。** 9 ms 的中位数尚可忍受，95 ms 的尾部不行；
token 之间有停顿时，p90 会升到 85 ms。每个 token 一次往返，意味着尾部延迟会
直接被用户看到。在有投机批处理之前（见 §5），**Wi-Fi 下诚实的默认选项不是
切分，而是本地模型**；有线缆时自动启用线缆。

**2. 对放得下的模型做切分是减分，不是加分。** 不要提供"用桌面端加速手机上的
本地模型"：实测表明这样更慢。应当提供"运行本机没有的模型"。

**3. 手机的调频器让速度相差一倍。** 在同一台设备上、36 °C、完全没有降频的
情况下，同一个 worker 计算它负责的层段，有时用 22.6 ms，有时用 42.4 ms：
`schedutil` 不会为一个只计算几毫秒就阻塞在套接字上的任务提升大核频率（实测：
cpu4 停留在 2400 MHz 中的 691 MHz）。应对方法，按正当性从高到低：

- **ADPF `PerformanceHintManager`**（Android 12+）：`APerformanceHint_createSession(tids,
  target_ns)`，并且每个 token 调用一次 `reportActualWorkDuration`。游戏引擎
  正是这样向内核说明帧截止时间的。ABI 中**已经有 `cortiq_worker_tids()`**——
  正是它所需的输入。
- 前台服务：进程进入 `top-app` cpuset 并获得超大核。上面的所有数字都是在
  `adb shell` 中测得的，即**下限**；在应用内部应当更好。
- `CMF_NET_SPIN`（µs，默认 3000）——用有界自旋代替阻塞读取，带来 +9%。在大核
  上完全自旋带来 +80%，但那是在消耗电池；不适合作为一种模式。

**4. 不要改动线程数。** 线程池会按大核数量自行确定大小
（`cores_from_capacities`），这已经是最优：绑定到大核簇带来 +0.3%（噪声），
而让线程池使用全部 8 个核心则是 **−28%**（14.3 → 10.3 tok/s），因为 A55 核心
会在每个层屏障处拖后腿。在这里，`cortiq_set_threads` 只会让情况变糟。

**5. `f16` 会改变文本。** 线路开销减半，与本地输出的分歧大约从第 30 个 token
开始。如果应用承诺可复现性（回答缓存、回归测试）——使用 `f32`。

**6. token 以明文传输。** 用线缆尚可容忍，用别人的 Wi-Fi 则不行。在伴侣功能
面向公共网络发布之前，需要 TLS/noise；目前诚实的模式只有可信网络或线缆。

---

## 4a. 第一版之后新增的内容（wire v6 和 v7）

**`--peer-run-ahead K`——一次往返返回 K 个 token。** 在覆盖整个层栈的 `head`
模式下，协调器在 token 之间什么也不做，因此为每个 token 请求许可只是多一次
往返，别无益处。这**不是**投机：不需要草稿模型，输出逐 token 一致。手机经
Wi-Fi：

| K | bonsai 1.7B | 34.7B MoE |
|---|---|---|
| 1 | 13.3 tok/s | 6.5 |
| 4 | 17.5 | 11.5 |
| 8–16 | **23.6** (K=16) | **12.4** (K=8) |

代价是 token 成批到达：K=4 时约每 230–260 ms 一批，K=16 时肉眼可见地不连贯。
应用中合理的默认值是 Wi-Fi 下 K=4、线缆下 K=1，因为线缆往返只需 1.9 ms，没有
什么可省的。

**`--peer-prefill`——在对端预填充，状态传回本机。** 这是唯一一种线路完成任务
后就可以撤掉的模式：桌面端计算长 prompt，KV 传到手机上，之后对话在本地继续，
线缆可以拔掉。在 1800 token 的 prompt 上实测：

| | 时间 |
|---|---|
| 在手机上预填充 | 86.9 s |
| 在 Mac 上预填充 + 经 USB 传回状态（f16，206.5 MB，35 MB/s） | 10.3 + 5.9 = **16.3 s** |

完全按照承诺所说的方式做了验证：状态一到达，就杀掉 worker 和隧道——手机
独自完成了剩余的生成。状态的开销是**每个位置 224 KiB**（实测），因此这里由
`--net-dtype f16` 决定传输是否划算：经 Wi-Fi 传输同样的 206 MB 约需 14 s。

如果无法描述状态（q8 存储、Nyström 覆盖层、冻结列），导出会**拒绝**传输——
而不是传出一个看似合理、回答却不同的缓存。重要性计数器随状态一起传输：每个
注意力层都会累积它，没有它，远端之后会淘汰错误的位置。

**`cortiq peers`——发现。** 方案是"查找方发出询问，worker 以单播回答"，而不是
"worker 主动广播"。原因经过实测：从笔记本电脑发往 Android 手机的广播
**收不到**（无论 `255.255.255.255` 还是子网形式），而单播总能送达。手机经
Wi-Fi 在五秒内就能找到桌面端。回答中包含文件名、`dir_hash`、几何结构、线路
协议版本以及是否需要 token；**消息中不含 token 本身**，文件路径也不会外传。

## 4b. 手机对手机：实测中的障碍

协议不认识"桌面端"这个词：两部手机以完全相同的方式分担计算，扮演协调器/
worker 角色，从这个意义上说，该模式已经可用。障碍在别处，而且是经过验证的，
不是推测：

1. **它们找不到彼此。** 在手机上启动的 worker 会占用端口且不报错——但 Mac
   扫描时**看不到**它：手机收不到广播查询。通过显式地址，同一个 worker 能
   正常服务（已验证：split 24 时 5.4 tok/s）。因此，两部手机需要另一种相互
   认识的渠道：二维码（"服务器"界面已经会绘制——需要反向扫描）、Wi-Fi Direct
   或 BLE。
2. **run-ahead 帮不了它们。** 只有当协调器在 token 之间什么也不做、即对端持有
   整个层栈时，它才是正当的。而两部手机分担计算恰恰是为了容量——各自计算一半
   ——于是 token 之间总有对方的矩阵乘法。在这种拓扑下，治疗 Wi-Fi 尾部延迟的
   办法用不上，而尾部延迟还会翻倍：两个无线电都会休眠。
3. **两部手机都需要完整的文件。** 身份按整个文件的 `dir_hash` 校验，因此每部
   手机都要为了自己的一半存下全部 12.9 GB。这是主要的实际障碍，它不在协议中，
   而在于身份无法按层范围校验。
4. **基于已有测量的速度估计。** 手机 worker 经线缆在 38 ms 内完成 14 层；两部
   手机各算一半，再加上线路：乐观情况下每个 token 约 85–100 ms，考虑调频器和
   尾部延迟后，经 Wi-Fi 为 5–8 tok/s。单部手机运行同一模型可达 14.3。结论与
   其他地方相同：**只有当模型在任何一部手机上都放不下时，手机配对才有意义**。

如果着手去做，工作顺序是：先做（3）——按范围校验身份，让每部手机只保存自己
那一半文件；然后做（1）——二维码配对；最后才测量真实的配对，因为数字将决定
它是否进入产品。

## 5. 运行时目前还没有的功能

以免应用围绕不存在的功能来设计：

- **投机批处理。** 瘦客户端甚至不需要它——那里用的是 run-ahead（§4a）。它
  恰恰会在禁止 run-ahead 的拓扑中被需要：协调器持有自己的层时。那时需要草稿
  ——带 MTP 头的模型；此测试平台上的模型都是 `MTP: no`，而跳层（layer-skip）
  投机经实测已经无效。
- **按层范围的身份校验。** 两端都必须持有完整文件，尽管各自只计算自己的部分。
  对手机配对而言，这是主要障碍（§4b）。
- **mDNS。** 已有基于查询的 UDP 发现（§4a），它在"手机查找桌面端"这个方向上
  可用。完整的 mDNS 只在需要出现在系统扫描器中时才有必要，而且会引入依赖。
- **`serve --peer` 中的 head 模式。** 采样器配置在 Assign 中只发送一次，而
  服务器的每个请求都有自己的配置；服务器路径仍不带 `head`。
- **加密。** token 仍以明文传输。
- **Android 上 GPU 带来的收益。** 它计算正确，但比 CPU 慢；启用后不再浪费
  几分钟，但也不加速。包含数字、反驳和工作清单的完整分析见
  `GPU_MOBILE.ru.md`（俄文）。
