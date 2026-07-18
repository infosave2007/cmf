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

#ifdef __cplusplus
}
#endif
#endif /* CORTIQ_H */
