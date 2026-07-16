//! Cortiq Server — OpenAI-compatible API + web management dashboard.

pub mod api;
pub mod dashboard;
pub mod openai;
pub mod streaming;

use axum::{routing::get, Json, Router};
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
    sem: Arc<Semaphore>,
}

/// A checked-out slot: holds both the concurrency permit and the
/// pipeline lock until dropped.
pub struct SlotGuard {
    _permit: OwnedSemaphorePermit,
    pub pipe: OwnedMutexGuard<Pipeline>,
}

impl PipelinePool {
    pub fn new(pipelines: Vec<Pipeline>) -> Self {
        assert!(!pipelines.is_empty(), "pipeline pool needs at least one slot");
        let sem = Arc::new(Semaphore::new(pipelines.len()));
        Self {
            slots: pipelines.into_iter().map(|p| Arc::new(Mutex::new(p))).collect(),
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
        for s in &self.slots {
            if let Ok(pipe) = s.clone().try_lock_owned() {
                return SlotGuard { _permit: permit, pipe };
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
}

/// Liveness probe — returns 200 as soon as the server is accepting
/// connections. Used by process managers that embed `cortiq serve` (e.g.
/// a gateway spawning it as a local model server) to know when it is ready.
async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
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
