# cortiq-ffi — embed the CMF runtime in your app

A C ABI over the CMF engine: load a `.cmf` file, stream tokens through a
callback, free. One call at a time per handle (internally serialized);
no panics cross the boundary; errors are readable strings.

The full surface is four functions — see [`include/cortiq.h`](include/cortiq.h).

## Android (JNI)

Build the shared library with [cargo-ndk](https://github.com/bbqsrc/cargo-ndk):

```sh
cargo install cargo-ndk
rustup target add aarch64-linux-android
cargo ndk -t arm64-v8a -o app/src/main/jniLibs build --release -p cortiq-ffi
# → app/src/main/jniLibs/arm64-v8a/libcortiq_ffi.so
```

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
    @JvmStatic external fun free(handle: Long)
}

// usage — run on a background thread, stream into your UI:
val h = Cortiq.load(File(filesDir, "model.cmf").absolutePath)
Cortiq.chat(h, "Привет!", 256) { token -> appendToUi(token); true }
Cortiq.free(h)
```

Notes for phones:
- keep the `.cmf` on storage — it is memory-mapped, RSS stays near the
  file size and load is instant;
- q4_tiled is the best size/speed point for mobile;
- the engine picks the big cores by itself on big.LITTLE (override with
  the `CMF_THREADS` env if you experiment);
- generation must not run on the main thread; the token callback fires
  on the calling thread.

## Desktop / iOS

The same four functions work anywhere: link the `staticlib`/`cdylib`
and include `cortiq.h`. iOS builds with the usual
`aarch64-apple-ios` target (staticlib + your Swift bridging header).
