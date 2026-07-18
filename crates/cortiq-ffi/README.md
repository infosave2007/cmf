# cortiq-ffi — embed the CMF runtime in your app

A C ABI over the CMF engine: load a `.cmf` file, stream tokens through a
callback, free. One call at a time per handle (internally serialized);
no panics cross the boundary; errors are readable strings.

The surface — see [`include/cortiq.h`](include/cortiq.h):

| function | what it does |
|---|---|
| `cortiq_load(path)` | open a `.cmf` (mmap), returns a handle |
| `cortiq_chat(h, prompt, max, cb, user)` | one user turn through the chat template |
| `cortiq_chat_messages(h, json, max, cb, user)` | multi-turn: `[{"role","content"},…]` through the template |
| `cortiq_complete(h, prompt, max, cb, user)` | raw completion, no template |
| `cortiq_set_options(h, json)` | sampler options, partial JSON: `temperature`, `top_p`, `top_k`, `repetition_penalty`, `min_p`, `seed`, `greedy` |
| `cortiq_free(h)` | release |
| `cortiq_last_error()` / `cortiq_version()` | diagnostics |

```c
cortiq_set_options(h, "{\"temperature\": 0.2, \"top_p\": 0.95}");   // sticky per handle
cortiq_chat_messages(h,
    "[{\"role\": \"system\", \"content\": \"Be brief.\"},"
    " {\"role\": \"user\", \"content\": \"Hi!\"}]",
    256, cb, user);
```

Absent option keys keep their current values (defaults: 0.7 / 0.9 / 40 /
1.1 / 0.05 / random seed); `"greedy": true` pins temperature to 0.
History for `cortiq_chat_messages` is the caller's: pass the full
conversation each call — the engine is stateless between calls.

## Android (JNI)

Build the shared library with [cargo-ndk](https://github.com/bbqsrc/cargo-ndk):

```sh
cargo install cargo-ndk
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 \
    -o app/src/main/jniLibs build --release -p cortiq-ffi
# → app/src/main/jniLibs/{arm64-v8a,armeabi-v7a,x86_64}/libcortiq_ffi.so
```

Or skip the toolchain: every release attaches prebuilt
`libcortiq-ffi-<target>.tar.gz` for all three ABIs — unpack into the
matching `jniLibs/` directory. arm64-v8a is the fast path (NEON SDOT
blocking, batched attention); armeabi-v7a and x86_64 run the portable
kernels — fine for old phones and emulators, don't benchmark on them.

Minimal JNI shim (`cortiq_jni.c`, compile into your app's native lib or
a tiny CMake target linking `libcortiq_ffi.so`):

```c
#include <jni.h>
#include "cortiq.h"

typedef struct { JNIEnv *env; jobject cb; jmethodID onToken; } StreamCtx;

static bool jni_token_cb(const char *token, void *user) {
    StreamCtx *s = (StreamCtx *)user;
    jstring t = (*s->env)->NewStringUTF(s->env, token);
    jboolean cont = (*s->env)->CallBooleanMethod(s->env, s->cb, s->onToken, t);
    (*s->env)->DeleteLocalRef(s->env, t);
    return cont == JNI_TRUE;
}

JNIEXPORT jlong JNICALL
Java_com_example_Cortiq_load(JNIEnv *env, jclass cls, jstring path) {
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    void *h = cortiq_load(p);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return (jlong)h;
}

JNIEXPORT jint JNICALL
Java_com_example_Cortiq_chat(JNIEnv *env, jclass cls, jlong h,
                             jstring prompt, jint maxTokens, jobject cb) {
    jclass cbCls = (*env)->GetObjectClass(env, cb);
    jmethodID onToken =
        (*env)->GetMethodID(env, cbCls, "onToken", "(Ljava/lang/String;)Z");
    StreamCtx s = { env, cb, onToken };
    const char *p = (*env)->GetStringUTFChars(env, prompt, NULL);
    jint n = cortiq_chat((void *)h, p, (uint32_t)maxTokens, jni_token_cb, &s);
    (*env)->ReleaseStringUTFChars(env, prompt, p);
    return n;
}

JNIEXPORT jint JNICALL
Java_com_example_Cortiq_chatMessages(JNIEnv *env, jclass cls, jlong h,
                                     jstring msgs, jint maxTokens, jobject cb) {
    jclass cbCls = (*env)->GetObjectClass(env, cb);
    jmethodID onToken =
        (*env)->GetMethodID(env, cbCls, "onToken", "(Ljava/lang/String;)Z");
    StreamCtx s = { env, cb, onToken };
    const char *m = (*env)->GetStringUTFChars(env, msgs, NULL);
    jint n = cortiq_chat_messages((void *)h, m, (uint32_t)maxTokens,
                                  jni_token_cb, &s);
    (*env)->ReleaseStringUTFChars(env, msgs, m);
    return n;
}

JNIEXPORT jint JNICALL
Java_com_example_Cortiq_setOptions(JNIEnv *env, jclass cls, jlong h,
                                   jstring opts) {
    const char *o = (*env)->GetStringUTFChars(env, opts, NULL);
    jint r = cortiq_set_options((void *)h, o);
    (*env)->ReleaseStringUTFChars(env, opts, o);
    return r;
}

JNIEXPORT void JNICALL
Java_com_example_Cortiq_free(JNIEnv *env, jclass cls, jlong h) {
    cortiq_free((void *)h);
}
```

Kotlin side:

```kotlin
object Cortiq {
    init { System.loadLibrary("cortiq_jni") }  // your shim, linked to libcortiq_ffi

    fun interface TokenListener { fun onToken(token: String): Boolean }

    @JvmStatic external fun load(path: String): Long
    @JvmStatic external fun chat(handle: Long, prompt: String,
                                 maxTokens: Int, cb: TokenListener): Int
    @JvmStatic external fun chatMessages(handle: Long, messagesJson: String,
                                         maxTokens: Int, cb: TokenListener): Int
    @JvmStatic external fun setOptions(handle: Long, optionsJson: String): Int
    @JvmStatic external fun free(handle: Long)
}

// usage — run on a background thread, stream into your UI:
val h = Cortiq.load(File(filesDir, "model.cmf").absolutePath)
Cortiq.setOptions(h, """{"temperature": 0.2}""")
Cortiq.chat(h, "Привет!", 256) { token -> appendToUi(token); true }
// multi-turn: serialize the whole history each call
Cortiq.chatMessages(h, gson.toJson(history), 256) { t -> appendToUi(t); true }
Cortiq.free(h)
```

Notes for phones:
- keep the `.cmf` on storage — it is memory-mapped, RSS stays near the
  file size and load is instant;
- q4_tiled is the best size/speed point for mobile;
- the engine picks its cores from the kernel's capacity table: on real
  big.LITTLE it takes the big cluster, on clock-binned same-µarch parts
  (JLQ JR510: 8×A55 as 4×2.0 + 4×1.5 GHz) it takes ALL of them —
  override with `CMF_THREADS` if you experiment;
- generation must not run on the main thread; the token callback fires
  on the calling thread.

Sizing for low-RAM devices (4 GB class, e.g. JR510 tablets): the model
file should stay well under free RAM or the page cache thrashes eMMC —
comfortable is ≤1 GB: Bonsai-1.7B q1 (334 MB), a 0.5–1.5B q8/q4t. A
27B q1 (4.8 GB) does NOT fit in 4 GB — it will run, at well under 1
tok/s, from storage. On all-A55 silicon expect single-digit tok/s for
a 1.7B: A55 is in-order with one 128-bit NEON pipe — measure on the
device, don't extrapolate from a flagship.

The arm64 and x86_64 `.so` ship with the Vulkan backend (Mali/Adreno).
It is opt-in: set `CMF_GPU=1` in the process environment *before*
`cortiq_load` (Kotlin: `android.system.Os.setenv("CMF_GPU", "1", true)`)
— a runtime probe then measures GPU vs your CPU per op class and keeps
whichever wins, so enabling it on a weak Mali is safe.

## iOS

Releases attach `libcortiq-ffi-aarch64-apple-ios.tar.gz` with the
static `libcortiq_ffi.a` — link it into the app binary (Xcode: add to
"Link Binary With Libraries", plus a bridging header including
`cortiq.h`). Flutter/Dart then reaches the symbols with
`DynamicLibrary.process()` — no separate .framework needed. To build it
yourself:

```sh
rustup target add aarch64-apple-ios
cargo build --release -p cortiq-ffi --target aarch64-apple-ios
# → target/aarch64-apple-ios/release/libcortiq_ffi.a
```

## Desktop

The same functions work anywhere: link the `staticlib`/`cdylib` and
include `cortiq.h`.
