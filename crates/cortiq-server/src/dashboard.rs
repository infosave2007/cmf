//! Web management dashboard — served as embedded static content.

use crate::AppState;
use axum::{
    response::Html,
    routing::get,
    Router,
};
use std::sync::Arc;

/// Register dashboard routes.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(serve_dashboard))
        .route("/dashboard", get(serve_dashboard))
}

/// Serve the embedded dashboard HTML.
async fn serve_dashboard() -> Html<&'static str> {
    Html(include_str!("../static/dashboard.html"))
}
