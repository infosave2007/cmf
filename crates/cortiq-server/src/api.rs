//! Cortiq extension API endpoints.

use crate::AppState;
use axum::{
    extract::State,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Register Cortiq extension routes.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/cortiq/status", get(get_status))
        .route("/v1/cortiq/masks", get(list_masks))
        .route("/v1/cortiq/switch", post(switch_task))
}

// ─── Status ──────────────────────────────────────────────

async fn get_status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let status = state.runtime.status().await;
    Json(serde_json::to_value(status).unwrap_or_default())
}

// ─── Masks ───────────────────────────────────────────────

#[derive(Serialize)]
struct MaskListResponse {
    masks: Vec<MaskInfo>,
}

#[derive(Serialize)]
struct MaskInfo {
    task_id: u32,
    name: String,
    sparsity: f32,
    /// Held-out quality value; null = not measured (never a declaration).
    quality_score: Option<f32>,
    quality_metric: Option<String>,
    active_layers: usize,
    active_neurons_avg: f64,
    has_hot_pack: bool,
}

async fn list_masks(State(state): State<Arc<AppState>>) -> Json<MaskListResponse> {
    let masks: Vec<MaskInfo> = state
        .runtime
        .masks()
        .masks
        .iter()
        .map(|m| MaskInfo {
            task_id: m.task_id,
            name: m.name.clone(),
            sparsity: m.sparsity,
            quality_score: m.quality.as_ref().map(|q| q.value),
            quality_metric: m.quality.as_ref().map(|q| q.metric.clone()),
            active_layers: m.active_layer_count(),
            active_neurons_avg: m.avg_active_neurons(),
            has_hot_pack: m.has_hot_pack,
        })
        .collect();

    Json(MaskListResponse { masks })
}

// ─── Task Switch ─────────────────────────────────────────

#[derive(Deserialize)]
struct SwitchRequest {
    task: String,
}

async fn switch_task(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SwitchRequest>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    match state.runtime.switch_task(&req.task).await {
        Ok(result) => Ok(Json(serde_json::to_value(result).unwrap_or_default())),
        Err(e) => {
            tracing::error!("Task switch failed: {}", e);
            Err(axum::http::StatusCode::NOT_FOUND)
        }
    }
}
