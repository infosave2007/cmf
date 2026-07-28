/* cortiq-ffi — C ABI over the CMF runtime.
 *
 * Threading: one call at a time per handle (internally serialized);
 * calls may come from any thread. Streaming callbacks fire
 * synchronously on the calling thread.
 * Errors: functions return NULL / -1; cortiq_last_error() describes the
 * most recent failure on the calling thread.
 */
#ifndef CORTIQ_H
#define CORTIQ_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Engine version, static UTF-8. */
const char *cortiq_version(void);

/* Most recent failure on this thread (UTF-8, valid until next failure). */
const char *cortiq_last_error(void);

/* Open a .cmf file (memory-mapped — keep it on storage). NULL on error. */
void *cortiq_load(const char *path);

/* Globally enable or disable the discrete GPU graph. Must be called before load. */
void cortiq_set_gpu(bool enable);

/* True when this build has a working GPU backend and the device can
 * bring an adapter up (Vulkan on Android, Metal on iOS/macOS). A
 * CPU-only library returns false while cortiq_set_gpu still accepts
 * the flag. */
bool cortiq_gpu_available(void);

/* Worker-pool size from the embedder (instead of the process-wide
 * CMF_THREADS environment variable). 0 = automatic. Call before
 * cortiq_load. */
void cortiq_set_threads(int32_t n);

/* Kernel thread ids of the current worker pool (Android/Linux; for
 * ADPF work-duration reporting). Copies up to cap ids into out and
 * returns the total count; 0 before a load or on other platforms. */
int32_t cortiq_worker_tids(int32_t *out, int32_t cap);

/* Cancel the generation currently running on this handle (thread-safe;
 * cortiq_chat* blocks its caller, so call this from another thread).
 * The run finishes with finish_reason "cancelled". */
void cortiq_cancel(void *handle);

/* Release a handle. NULL is a no-op. */
void cortiq_free(void *handle);

/* Streaming callback: token is NUL-terminated UTF-8, valid only during
 * the call; return true to continue generating, false to stop early. */
typedef bool (*cortiq_token_cb)(const char *token, void *user);

/* One chat turn through the file's own chat template (models without a
 * template fall back to plain completion). Returns generated-token
 * count, or -1. cb may be NULL (generate without streaming). */
int32_t cortiq_chat(void *handle, const char *prompt, uint32_t max_tokens,
                    cortiq_token_cb cb, void *user);

/* Raw completion: the prompt reaches the model verbatim (plus the
 * tokenizer's BOS contract). Same contract as cortiq_chat. */
int32_t cortiq_complete(void *handle, const char *prompt,
                        uint32_t max_tokens, cortiq_token_cb cb, void *user);

/* Multi-turn chat: messages_json is [{"role": "...", "content": "..."},
 * ...] rendered through the file's own chat template (roles the
 * template knows — typically system / user / assistant). Same
 * streaming/return contract as cortiq_chat. */
int32_t cortiq_chat_messages(void *handle, const char *messages_json,
                             uint32_t max_tokens, cortiq_token_cb cb,
                             void *user);

/* Partial sampler options as JSON — absent keys keep their current
 * values; applies to every later generate on this handle. Keys:
 * temperature, top_p, top_k, repetition_penalty, min_p, seed,
 * greedy (true = argmax). Defaults: 0.7 / 0.9 / 40 / 1.1 / 0.05 /
 * random. Returns 0, or -1. */
int32_t cortiq_set_options(void *handle, const char *options_json);

#ifdef __cplusplus
}
#endif
#endif /* CORTIQ_H */
