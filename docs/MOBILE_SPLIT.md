# The phone in a network split: what to add to the app

*Languages: **English** · [Русский](MOBILE_SPLIT.ru.md) · [中文](MOBILE_SPLIT.zh.md)*

Status: the ABI is ready and measured in 0.5.70. App-side work has not started.
Related documents: `MULTI_DEVICE_ROADMAP.ru.md` (§3.5, Phase 4b; in Russian),
`CMF_V2_SPEC.md`, the header `crates/cortiq-ffi/include/cortiq.h`.

---

## 0. What exactly was measured — and why the role matters more than the split

Stand: Xiaomi 12 Lite (SD778G, 4×A78 2.4 GHz + 4×A55, 7.3 GB RAM, about 2 of
them free) over USB-C and on the same 5 GHz 802.11ax Wi-Fi as a MacBook M4 24 GB.
A real `cortiq worker` in both directions, not an emulation.

**A layer split does not speed up a token.** A token goes through the layers
sequentially; distribution buys memory, not parallelism. Measured in both
directions and on both transports: any configuration in which both sides compute
is slower than the fastest side alone. Bonsai 1.7B: Mac 28.6 tok/s, phone 14.3
(steady), half and half over USB — 14.0.

Hence **four roles**, and the app should choose a role, not a "load
percentage":

| role | what it gives | measured |
|---|---|---|
| **Thin client** — the desktop holds everything, the phone only draws | a model that cannot live on the phone | a 34.7B MoE on a phone with 2 GB free: **16.3 tok/s** |
| **Prefill offload** — the desktop computes a long prompt | time to first token | 1800 positions: 86.9 s → **11.3 s (7.7×)** |
| **Memory extender** — the phone holds the tail of the desktop's layers | a model larger than the desktop | a win only when the model does not fit; otherwise a pure loss |
| **Replica** — the phone serves the whole request | throughput | this is today's "Server" screen, and it is already right |

The transport is ranked by its **tail**, not by megabits. A 4 KB round trip
(exactly one hidden state at hidden 2048):

| path | p50 | p90 | p99 |
|---|---|---|---|
| USB | **1.89 ms** | 2.19 | 2.94 |
| Wi-Fi 5 GHz ax, −53 dBm, 680 Mbit/s link | 8.95 | 11.39 | **94.81** |
| Wi-Fi with a 100 ms pause between tokens | 9.56 | **85.14** | 189.8 |

One round trip per token means the user sees the p99. The cable beats excellent
Wi-Fi fivefold at the median and thirtyfold at the tail.

---

## 1. What appeared in the ABI (0.5.70)

Three functions, all additive. An app that does not call them works as
before.

```c
int32_t     cortiq_worker_start(const char *config_json);
int32_t     cortiq_set_peer(const char *config_json);
const char *cortiq_peer_stats(void);
```

**`cortiq_worker_start`** — `cortiq worker` as a call: the phone lends a range
of layers of its copy of the model to someone else's coordinator. The listener
lives on a background thread.

```json
{"model": "/data/.../model.cmf", "listen": "0.0.0.0:9911", "token": "secret"}
```

The port is bound and the file is checked **on the caller's thread**, so a busy
port or a missing file is `-1` plus a message in `cortiq_last_error()` that can be
shown to the user, not a silently dead thread. The token is **mandatory** for
any address other than loopback: without it the worker refuses to listen.

**`cortiq_set_peer`** — the reverse: every subsequent generation goes through a
peer that holds **the same** file.

```json
{"addr": "192.168.1.5:9911", "token": "secret",
 "split": 0, "dtype": "f16", "head": true}
```

- `split` — the first layer the peer computes. `0` = all of them, and then this
  side keeps only the tokenizer. The default is half the stack.
- `head` — the peer also owns `lm_head` with the sampler and answers with a
  **token id**.
- `dtype` — `f32` reproduces the local text byte for byte, `f16` halves the
  wire and legitimately changes the text.

`null` or `{}` clears it and returns to local mode. The connection is brought up
lazily, on the first generation — an unreachable peer fails there, with a message.

**`cortiq_peer_stats`** — what the peer is worth **right now**, as JSON:

```json
{"thermal_mc":34300,"powered":null,"cpu_khz_cur":691200,
 "cpu_khz_max":2400000,"mem_avail_kb":1467000,"threads":4,
 "platform":"android/aarch64"}
```

`null` means "the platform does not report it", not zero. Do not substitute
zeros there: a scheduler that reads 0 MHz will park a live node.

---

## 2. The `head` flag — the cheapest speedup there is here

The head (`final norm` + `lm_head` + sampler) **does not shrink**, however many
layers you move away. On the phone it cost 29 ms of a 73 ms token — so the
phone client was bound by its own head regardless of the desktop's speed. With
`head: true` the head moves to the peer, and the token id travels back.

Phone coordinator + Mac worker, all layers on the Mac, over USB:

| model | `head: false` | `head: true` | |
|---|---|---|---|
| bonsai 1.7B | 12.6–13.0 tok/s | **26.0–27.3** | 2.06× |
| Qwen3.6 35B-A3B q2tp (34.7B) | 9.1–9.4 | **15.7–16.3** | 1.72× |

In this mode the wire per token is **16 bytes** (12 out, 4 back) instead of
4 KB, and the `net:` line shows "99% of decode wall": the phone only waits and
detokenizes.

Why we could not move **only** the head, without the sampler: the logits of these
models are 151,669 numbers, 300 KB per token in f16. At the measured Wi-Fi
bandwidth of 14.6 MB/s that is 20 ms per token — more than it saves. It is the
id that has to travel.

**Parity:** with the same split and the same backend on both sides,
`head: true` gives byte-for-byte the same text as `head: false` — both in greedy
and in sampling with a fixed seed (the sampler config travels to the peer in
Assign; otherwise the default temperature on that side would silently rewrite
the answer). Caveat: the mode moves the head to the **peer's device**, so if the
sides are on different backends (phone CPU, desktop Metal), the text may
legitimately change — the same rule as for any cross-device split.

---

## 3. What to do in the app, step by step

### Step 1. Rebuild the native library

The build matrix is NOT the same across ABIs, and either peculiarity silently
breaks the app if you miss it.

```sh
# 64-bit ABIs — with the gpu feature. It is NOT a default: a build without it gives
# a library half the size and an app without Vulkan, without a single error.
cargo ndk -t arm64-v8a -t x86_64 --platform 26 \
  build --release -p cortiq-ffi --features gpu

# armeabi-v7a — without gpu (that is how the library shipped in the app is built) AND with
# 16 KB pages: by default the linker gives 0x1000 alignment, and such a
# library will not load on Android 15 and will not pass Play.
RUSTFLAGS="-C link-arg=-Wl,-z,max-page-size=16384" \
  cargo ndk -t armeabi-v7a --platform 26 build --release -p cortiq-ffi

# iOS — also with gpu, and both targets: device and simulator (arm64)
cargo build --release -p cortiq-ffi --target aarch64-apple-ios --features gpu
cargo build --release -p cortiq-ffi --target aarch64-apple-ios-sim --features gpu
```

Check BEFORE copying — the three symbols are in place and the alignment is
0x4000 for all three ABIs (`NDK/toolchains/llvm/prebuilt/*/bin`):

```sh
llvm-nm -D libcortiq_ffi.so | grep -c "cortiq_\(worker_start\|set_peer\|peer_stats\)"   # 3
llvm-readelf -l libcortiq_ffi.so | awk '/LOAD/{print $NF}' | sort -u        # 0x4000
```

The sizes are a check too: arm64 ≈ 14.1 MB and x86_64 ≈ 14.5 MB (with gpu),
v7a ≈ 4.5 MB (without), both iOS libraries ≈ 88.7 MB (with gpu; 53.6 without
it — if that is what you got, the feature was lost).

**The symbols in the iOS archive are invisible to plain `nm`, and that is not a
breakage.** The release profile builds with `lto = "thin"`, so the archive
members are LLVM bitcode, and `nm` returns nothing even for `_cortiq_version`.
Check by linking instead:

```sh
printf 'int32_t cortiq_set_peer(const char*);\nint main(){return cortiq_set_peer("{}");}\n' > probe.c
xcrun -sdk iphoneos clang -target arm64-apple-ios13.0 probe.c libcortiq_ffi.a \
  -framework Metal -framework Foundation -framework CoreFoundation \
  -framework CoreGraphics -framework QuartzCore -framework IOSurface -o probe
```

`CoreGraphics` in this list is mandatory: wgpu's Metal surface pulls in
`CGColorSpace*`, and without it the link fails on those, not on our symbols.

Copy `libcortiq_ffi.so` to `android/app/src/main/jniLibs/<abi>/` and update
`native/cortiq_ffi.h` from `crates/cortiq-ffi/include/cortiq.h`.

**Both sides must speak the same protocol version.** The handshake compares
`WIRE_VERSION` (currently 7) and refuses with the text "wire version N ≠ 7 —
upgrade one side". This is a feature: an incompatibility shows up at once, not
as garbage in the output.

### Step 2. Bindings

In the style of the existing `lib/data/services/inference/native_engine.dart`:

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

/// Routes subsequent generations through the desktop. `split: 0` + `head: true`
/// is the thin-client mode: the phone only tokenizes.
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

/// Return to the local model.
void useLocal() => _setPeer(ffi.nullptr);

Map<String, dynamic> peerStats() =>
    jsonDecode(_peerStats().toDartString()) as Map<String, dynamic>;
```

`cortiq_set_peer` does not block and does not touch the network — the connection
comes up on the first generation. Show the error from the same place where
generation errors are shown today: through `cortiq_last_error()`.

### Step 3. The "Companion" screen

The minimum that is already useful:

1. An address and token field (or a QR code — the "Server" screen already draws
   one; here the reverse is needed: the desktop shows it, the phone scans it).
2. A role switch: **"Compute here"** / **"Compute on the desktop"**.
3. A status line, honest and to the point — what is computed where and over
   which wire.
4. A "Check" button — one short request; if the peer is not up, show the error
   text, not a spinner.

Model requirement: **both sides must hold the same `.cmf`**.
Identity is checked by `dir_hash` at the handshake, and a foreign file is
rejected — a chimera is never assembled. In the UI this means: match the
desktop's model list against the local list instead of offering to "connect to
anything".

The file is needed on the phone even with `split: 0` — the tokenizer and the
chat template are read from it. On disk this is the full model size, in memory
only the touched pages (34.7B q2tp: 12.9 GB on disk, runs with 2 GB of free RAM).

### Step 4. The worker role — foreground only

`cortiq_worker_start` brings up the listener, but the phone has to stay alive and
out of doze: a foreground service + a wake lock. The app already has
`core/util/foreground_task.dart` and `core/util/keep_awake.dart` — the same path
as the "Server" mode.

### Step 5. Polling the state

Call `cortiq_peer_stats()` between turns (not in the decode loop) and show
`cpu_khz_cur / cpu_khz_max`. Below ~50% during active decode means the governor
on that side has not woken up — and that can be fixed, see trap 3.

---

## 4. Traps — each one cost a measurement

**1. Wi-Fi is cured only by batching.** A 9 ms median is tolerable, a 95 ms tail
is not; with pauses between tokens the p90 creeps up to 85 ms. One round trip
per token means the tail hits the user's eyes. Until there is a speculative
batch (see §5), **the honest default for Wi-Fi is not a split but the local
model**; switch the cable on automatically when it is there.

**2. Splitting a model that fits is a minus, not a plus.** Do not offer to "speed
up the phone's local model with the desktop": it was measured to be slower.
Offer to "run a model that is not here".

**3. The phone's governor costs a factor of two.** On the same hardware at 36 °C,
with no throttling at all, one and the same worker served its range in 22.6 ms
one time and 42.4 ms another: `schedutil` does not ramp up the big cores for a
task that computes for a few milliseconds and blocks on a socket (measured:
cpu4 stayed at 691 of 2400 MHz). What to do, in decreasing order of
legitimacy:

- **ADPF `PerformanceHintManager`** (Android 12+): `APerformanceHint_createSession(tids,
  target_ns)` and `reportActualWorkDuration` per token. This is exactly how game
  engines tell the kernel about a frame deadline. The ABI **already has
  `cortiq_worker_tids()`** — exactly the input this needs.
- A foreground service: the process lands in the `top-app` cpuset and gets the
  prime core. All the numbers above were taken from `adb shell`, so they are a
  **lower bound**; inside the app it should be better.
- `CMF_NET_SPIN` (µs, default 3000) — a bounded spin instead of a
  blocking read, gave +9%. A full spinner on the big cores gave +80%, but that
  costs battery; it is no good as a mode.

**4. Do not touch the thread count.** The pool sizes itself to the big cores
(`cores_from_capacities`), and that is already the optimum: pinning to the big
cluster gives +0.3% (noise), and letting the pool loose on all 8 cores gives
**−28%** (14.3 → 10.3 tok/s), because an A55 core becomes the straggler at every
layer barrier. `cortiq_set_threads` is a way to make things worse here.

**5. `f16` changes the text.** The wire is half as expensive, and the divergence
from local output starts at about the 30th token. If the app promises
reproducibility (an answer cache, regression tests) — use `f32`.

**6. The token travels in plain text.** Tolerable over a cable, not over someone
else's Wi-Fi. Before the companion ships for public networks, TLS/noise is
needed; for now the honest mode is a trusted network or a cable only.

---

## 4a. What was added after the first edition (wire v6 and v7)

**`--peer-run-ahead K` — K tokens per round trip.** In `head` mode over the
whole stack the coordinator does nothing between tokens, so asking permission
for each one is an extra round trip and nothing more. This is NOT speculation:
no draft model is needed, and the output matches token for token. Phone over
Wi-Fi:

| K | bonsai 1.7B | 34.7B MoE |
|---|---|---|
| 1 | 13.3 tok/s | 6.5 |
| 4 | 17.5 | 11.5 |
| 8–16 | **23.6** (K=16) | **12.4** (K=8) |

The price: tokens arrive in bursts — at K=4 a burst every ~230–260 ms, at K=16
the jerkiness is visible to the eye. A sensible default in the app is K=4 over
Wi-Fi and K=1 over a cable, where a round trip costs 1.9 ms and there is nothing
to save.

**`--peer-prefill` — prefill on the peer, the state comes home.** The only mode
in which the wire does its job and leaves: the desktop computes a long prompt,
the KV arrives on the phone, the conversation then continues locally, and the
cable can be pulled out. Measured on an 1800-token prompt:

| | time |
|---|---|
| prefill on the phone | 86.9 s |
| prefill on the Mac + state over USB (f16, 206.5 MB, 35 MB/s) | 10.3 + 5.9 = **16.3 s** |

Checked exactly the way the promise reads: the worker and the tunnel were killed
the moment the state arrived — the phone finished generating on its own. The
state costs **224 KiB per position** (measured), so `--net-dtype f16` decides
here whether the transfer pays off: over Wi-Fi the same 206 MB take ~14 s.

The export **refuses** to go if it cannot describe the state (q8 storage, a
Nyström overlay, frozen columns) — instead of a plausible cache that
answers differently. The importance counter travels with the state: every
attention layer accumulates it, and without it the far side would later evict
the wrong positions.

**`cortiq peers` — discovery.** The scheme is "the seeker asks, the worker
answers by unicast", not "the worker shouts". The reason was measured: a
broadcast from a laptop to an Android phone **does not arrive** (neither
`255.255.255.255` nor the subnet form), while unicast always arrives. The phone
finds the desktop over Wi-Fi in five seconds. The answer carries the file name,
`dir_hash`, the geometry, the wire version and whether a token is needed; **the
token itself is not in the message**, and the file path does not leave either.

## 4b. Phone to phone: what gets in the way, by measurement

The protocol does not know the word "desktop": two phones share in exactly the
same way, with coordinator/worker roles, and in that sense the mode already
works. What gets in the way is something else, and it was checked, not
assumed:

1. **They will not find each other.** A worker brought up on a phone takes the
   port and does not complain — but the Mac **does not see** it in a scan: the
   phone does not receive the broadcast query. At an explicit address the same
   worker serves normally (checked: 5.4 tok/s at split 24). So a pair of phones
   needs another way to get introduced: a QR code (the "Server" screen already
   draws one — the reverse scan is needed), Wi-Fi Direct or BLE.
2. **Run-ahead will not help them.** It is legitimate only when the coordinator
   does nothing between tokens, i.e. when the peer holds the whole stack. But
   two phones share precisely for capacity — each computes its half — and then
   there is always the other side's matmul between tokens. The cure for the
   Wi-Fi tail does not apply in this topology, and the tail doubles: both radios
   sleep.
3. **Both need the whole file.** Identity is checked by the `dir_hash` of the
   whole file, so each phone stores all 12.9 GB for the sake of its half. This
   is the main practical brake, and it is not in the protocol but in the fact
   that identity cannot be checked per layer range.
4. **A speed estimate from what is already measured.** A phone worker serves
   14 layers in 38 ms over a cable; in a pair each computes its half, plus the
   wire: a token comes out at ~85–100 ms in the optimistic case, i.e. 5–8 tok/s
   over Wi-Fi allowing for the governor and the tail. One phone on the same
   model gives 14.3. The conclusion is the same as everywhere: **a pair of
   phones is justified only for a model that fits in neither of them**.

The order of work, if we take it on: first (3) — identity per range, so that
each phone holds its own half of the file; then (1) — QR pairing; and only then
measure a real pair, because the number will decide whether this goes into the
product.

## 5. What the runtime does not have yet

So that the app is not designed around something that does not exist:

- **A speculative batch.** The thin client does not even need it — run-ahead
  works there (§4a). It will be needed exactly in the topology where run-ahead is
  forbidden: when the coordinator holds its own layers. Then a draft is needed —
  a model with an MTP head; the models on this stand have `MTP: no`, and
  layer-skip speculation was measured and is dead.
- **Identity per layer range.** Both sides must hold the whole file, although
  each computes only its part. For a pair of phones this is the main brake (§4b).
- **mDNS.** There is UDP discovery by query (§4a), and it works in the
  "phone looks for the desktop" direction. Full mDNS is needed only to show up
  in system scanners, and it pulls in a dependency.
- **Head mode in `serve --peer`.** The sampler config travels once in Assign,
  whereas the server has its own per request; the server path stays without
  `head`.
- **Encryption.** The token still travels in plain text.
- **A benefit from the GPU on Android.** It computes correctly but loses to the
  CPU; switched on, it no longer costs minutes, but it does not speed things up
  either. The full analysis with numbers, refutations and a work list is in
  `GPU_MOBILE.ru.md` (in Russian).
