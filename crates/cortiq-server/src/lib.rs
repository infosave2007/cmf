//! Cortiq Server — OpenAI-compatible API + web management dashboard.

pub mod api;
pub mod dashboard;
pub mod openai;
pub mod streaming;

use axum::Router;
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

/// Build the full router with all endpoints.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .merge(openai::routes())
        .merge(api::routes())
        .merge(dashboard::routes())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
