//! perf-server - HTTP bridge exposing the `perf` library to the Python core.

use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use perf::{checksum, cosine_batch, normalize};

#[derive(Deserialize)]
struct CosineBatchReq {
    query: Vec<f32>,
    corpus: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct CosineBatchResp {
    scores: Vec<f32>,
    n: usize,
}

#[derive(Deserialize)]
struct ChecksumReq {
    text: String,
}

#[derive(Serialize)]
struct ChecksumResp {
    crc32: u32,
    bytes: usize,
}

#[derive(Deserialize)]
struct NormalizeReq {
    vector: Vec<f32>,
}

#[derive(Serialize)]
struct NormalizeResp {
    vector: Vec<f32>,
    norm: f32,
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    service: &'static str,
    version: &'static str,
}

async fn health() -> impl IntoResponse {
    Json(Health {
        status: "ok",
        service: "perf-server",
        version: env!("CARGO_PKG_VERSION"),
    })
}

async fn cosine_handler(Json(req): Json<CosineBatchReq>) -> impl IntoResponse {
    if req.corpus.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "empty corpus"})),
        )
            .into_response();
    }
    if let Some(bad) = req.corpus.iter().find(|r| r.len() != req.query.len()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "vector length mismatch",
                "expected": req.query.len(),
                "got": bad.len(),
            })),
        )
            .into_response();
    }
    let scores = cosine_batch(&req.query, &req.corpus);
    let n = scores.len();
    Json(CosineBatchResp { scores, n }).into_response()
}

async fn checksum_handler(Json(req): Json<ChecksumReq>) -> impl IntoResponse {
    let bytes = req.text.as_bytes();
    Json(ChecksumResp {
        crc32: checksum(bytes),
        bytes: bytes.len(),
    })
}

async fn normalize_handler(Json(mut req): Json<NormalizeReq>) -> impl IntoResponse {
    let norm = normalize(&mut req.vector);
    Json(NormalizeResp {
        vector: req.vector,
        norm,
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/cosine_batch", post(cosine_handler))
        .route("/checksum", post(checksum_handler))
        .route("/normalize", post(normalize_handler));

    let port: u16 = std::env::var("PERF_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7070);
    let host: String = std::env::var("PERF_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid PERF_HOST/PERF_PORT");

    eprintln!("perf-server listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind failed");
    axum::serve(listener, app).await.expect("server crashed");
}
