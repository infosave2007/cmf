//! Cortiq Server — OpenAI-compatible API + web management dashboard.

// Decision Vulkan adds a deep wgpu auto-trait graph to spawned server futures.
// This also covers the library's own async integration-style tests.
#![recursion_limit = "256"]

pub mod api;
pub mod dashboard;
pub mod decisions;
pub mod ood;
pub mod openai;
pub mod route;
pub mod streaming;
pub mod tool_calls;

pub use route::SkillRouter;

use axum::extract::State;
use axum::{Json, Router, routing::get};
use cortiq_engine::{CortiqRuntime, Pipeline};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard, OwnedSemaphorePermit, Semaphore};
use tower_http::cors::CorsLayer;

/// Fixed pool of pipeline slots over ONE shared mmap'd model (roadmap
/// §3 «serving полностью сериализован», этап 5.1): the weights are
/// zero-copy shared through `Arc<CmfModel>`, each slot owns its
/// KV-cache / recurrent state / sampler / workspace. A request checks a
/// slot out for the duration of one generation, so up to `slots`
/// requests decode CONCURRENTLY; excess requests queue fairly on the
/// semaphore. This is bounded-concurrency serving, not yet continuous
/// batching (этап 5.2+).
pub struct PipelinePool {
    slots: Vec<Arc<Mutex<Pipeline>>>,
    /// GPU each slot's weights live on (replica mode: slot i → card i).
    /// Empty = single-device, every slot on the process default.
    devices: Vec<usize>,
    sem: Arc<Semaphore>,
}

/// A checked-out slot: holds both the concurrency permit and the
/// pipeline lock until dropped.
pub struct SlotGuard {
    pub pipe: OwnedMutexGuard<Pipeline>,
    /// The card this slot's weights are on. The handler thread is
    /// pinned to it for the whole request — the engine resolves its
    /// device context (and therefore its weight cache) through that pin.
    pub device: usize,
    // Keep the permit after the mutex guard so drop glue unlocks the
    // pipeline before another waiter can acquire the permit.
    _permit: OwnedSemaphorePermit,
}

impl PipelinePool {
    pub fn new(pipelines: Vec<Pipeline>) -> Self {
        let n = pipelines.len();
        Self::with_devices(pipelines, vec![cortiq_engine::gpu::default_device(); n])
    }

    /// Replica mode: `devices[i]` is the card slot i was loaded on.
    pub fn with_devices(pipelines: Vec<Pipeline>, devices: Vec<usize>) -> Self {
        assert!(
            !pipelines.is_empty(),
            "pipeline pool needs at least one slot"
        );
        assert_eq!(
            pipelines.len(),
            devices.len(),
            "one device per slot: {} pipelines, {} devices",
            pipelines.len(),
            devices.len()
        );
        let sem = Arc::new(Semaphore::new(pipelines.len()));
        Self {
            slots: pipelines
                .into_iter()
                .map(|p| Arc::new(Mutex::new(p)))
                .collect(),
            devices,
            sem,
        }
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    /// Snapshot each lane's cache and recurrent-state allocation.  The
    /// locks are intentionally taken only for diagnostics; generation keeps
    /// its lane lock for the full request, so a snapshot waits for in-flight
    /// work and cannot observe a half-updated state.
    pub async fn slot_memory_bytes(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let pipe = slot.lock().await;
            out.push((
                pipe.kv_cache.attention_state_bytes(),
                pipe.kv_cache.recurrent_state_bytes(),
            ));
        }
        out
    }

    /// Wait for a free slot and check it out. With `permits == slots`,
    /// holding a permit guarantees the try_lock scan finds a free slot.
    pub async fn acquire(&self) -> SlotGuard {
        let permit = self
            .sem
            .clone()
            .acquire_owned()
            .await
            .expect("slot semaphore closed");
        for (i, s) in self.slots.iter().enumerate() {
            if let Ok(pipe) = s.clone().try_lock_owned() {
                let device = self.devices[i];
                // Pin the caller's thread: everything this request does
                // downstream — including the worker pool, which carries
                // the pin with each dispatch — addresses this card.
                cortiq_engine::gpu::set_current_device(device);
                return SlotGuard {
                    pipe,
                    device,
                    _permit: permit,
                };
            }
        }
        unreachable!("semaphore permit held but every slot is locked")
    }
}

/// Shared application state: runtime (masks, metrics), a tokenizer
/// handle that never blocks on generation, and the slot pool.
pub struct AppState {
    pub runtime: CortiqRuntime,
    pub tokenizer: Arc<cortiq_engine::tokenizer::Tokenizer>,
    pub slots: PipelinePool,
    /// Network pipeline-split worker (serve --peer). One worker holds one
    /// KV session, so peer mode runs with exactly one slot; the mutex is
    /// never contended (the slot semaphore already serializes) but keeps
    /// the type honest.
    pub remote: Option<Arc<std::sync::Mutex<cortiq_net::RemoteSegment>>>,
    /// Router v2 (`header.router`): per-request backbone-gated routing on
    /// the last user message, one lazily loaded lane per routed skill;
    /// `slots` is the backbone lane. `None` = every request runs `slots`.
    pub routing: Option<Arc<SkillRouter>>,
}

/// Liveness probe — returns 200 as soon as the server is accepting
/// connections. Used by process managers that embed `cortiq serve` (e.g.
/// a gateway spawning it as a local model server) to know when it is ready.
/// Also advertises the loaded model's capabilities so managers can route
/// capability-gated traffic (tool calling) without manual configuration:
/// tools are "supported" when the model's chat template has a tools branch.
async fn healthz(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let dsv41 = st.runtime.model().arch().deepseek_v41.is_some();
    let tools = dsv41
        || st
            .tokenizer
            .chat_template
            .as_deref()
            .map(|t| t.contains("tool"))
            .unwrap_or(false);
    Json(serde_json::json!({
        "status": "ok",
        "capabilities": {
            "tools": tools,
            "vision": dsv41,
            "reasoning_effort": dsv41,
            "dsml": dsv41
        }
    }))
}

/// Build the full router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .merge(openai::routes())
        .merge(api::routes())
        .merge(dashboard::routes())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
