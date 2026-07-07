//! Cortiq Server — OpenAI-compatible API + web management dashboard.

pub mod api;
pub mod dashboard;
pub mod openai;
pub mod streaming;

use axum::{routing::get, Json, Router};
use cortiq_engine::{CortiqRuntime, Pipeline};
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

/// Shared application state: runtime (masks, metrics) + the inference
/// pipeline behind a Mutex (single-sequence decode; requests queue).
pub struct AppState {
    pub runtime: CortiqRuntime,
    pub pipeline: Mutex<Pipeline>,
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
