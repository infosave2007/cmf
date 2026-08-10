//! Cortiq Server — OpenAI-compatible API + web management dashboard.

pub mod api;
pub mod dashboard;
pub mod openai;
pub mod streaming;

use axum::{Json, Router, routing::get};
use axum::extract::State;
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
}

/// Liveness probe — returns 200 as soon as the server is accepting
/// connections. Used by process managers that embed `cortiq serve` (e.g.
/// a gateway spawning it as a local model server) to know when it is ready.
/// Also advertises the loaded model's capabilities so managers can route
/// capability-gated traffic (tool calling) without manual configuration:
/// tools are "supported" when the model's chat template has a tools branch.
async fn healthz(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let tools = st
        .tokenizer
        .chat_template
        .as_deref()
        .map(|t| t.contains("tool"))
        .unwrap_or(false);
    Json(serde_json::json!({
        "status": "ok",
        "capabilities": { "tools": tools }
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
