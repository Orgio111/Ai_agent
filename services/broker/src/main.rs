use anyhow::Result;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::{net::SocketAddr, sync::Arc};
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod broker;
mod dlq;
mod event_store;
mod metrics;
mod queue;
mod websocket;

use broker::EventBroker;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
    let bind_addr = std::env::var("BROKER_ADDR").unwrap_or_else(|_| "0.0.0.0:8001".into());

    let broker = Arc::new(EventBroker::new(&redis_url).await?);

    // Spawn background workers
    let broker_clone = broker.clone();
    tokio::spawn(async move { broker_clone.run_dispatch_loop().await });

    let broker_clone = broker.clone();
    tokio::spawn(async move { broker_clone.run_dlq_requeue_loop().await });

    let broker_clone = broker.clone();
    tokio::spawn(async move { broker_clone.run_health_check_loop().await });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/publish", post(publish_handler))
        .route("/subscribe", post(subscribe_handler))
        .route("/cancel/:event_id", post(cancel_handler))
        .route("/dlq", get(dlq_list_handler))
        .route("/dlq/:event_id/retry", post(dlq_retry_handler))
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/events", get(event_history_handler))
        .layer(CorsLayer::permissive())
        .with_state(broker);

    let addr: SocketAddr = bind_addr.parse()?;
    info!("JARVIS Broker listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(broker): State<Arc<EventBroker>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket::handle_socket(socket, broker))
}

async fn publish_handler(
    State(broker): State<Arc<EventBroker>>,
    Json(event): Json<queue::Event>,
) -> impl IntoResponse {
    match broker.publish(event).await {
        Ok(id) => Json(serde_json::json!({ "event_id": id, "status": "queued" })),
        Err(e) => {
            warn!("Publish error: {e}");
            Json(serde_json::json!({ "error": e.to_string() }))
        }
    }
}

async fn subscribe_handler(
    State(broker): State<Arc<EventBroker>>,
    Json(req): Json<serde_json::Value>,
) -> impl IntoResponse {
    let topic = req["topic"].as_str().unwrap_or("*").to_string();
    let subscriber_id = req["subscriber_id"]
        .as_str()
        .unwrap_or("anon")
        .to_string();
    broker.register_subscriber(subscriber_id.clone(), topic.clone());
    Json(serde_json::json!({ "subscriber_id": subscriber_id, "topic": topic, "status": "subscribed" }))
}

async fn cancel_handler(
    State(broker): State<Arc<EventBroker>>,
    axum::extract::Path(event_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    broker.cancel_event(&event_id);
    Json(serde_json::json!({ "event_id": event_id, "status": "cancelled" }))
}

async fn dlq_list_handler(State(broker): State<Arc<EventBroker>>) -> impl IntoResponse {
    let items = broker.dlq_list();
    Json(serde_json::json!({ "dead_letter_queue": items }))
}

async fn dlq_retry_handler(
    State(broker): State<Arc<EventBroker>>,
    axum::extract::Path(event_id): axum::extract::Path<String>,
) -> impl IntoResponse {
    match broker.dlq_retry(&event_id).await {
        Ok(()) => Json(serde_json::json!({ "event_id": event_id, "status": "requeued" })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

async fn metrics_handler() -> impl IntoResponse {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let families = prometheus::gather();
    let mut buf = Vec::new();
    let _ = encoder.encode(&families, &mut buf);
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        buf,
    )
}

async fn health_handler(State(broker): State<Arc<EventBroker>>) -> impl IntoResponse {
    Json(broker.health_status().await)
}

async fn event_history_handler(State(broker): State<Arc<EventBroker>>) -> impl IntoResponse {
    let history = broker.event_history(100).await;
    Json(serde_json::json!({ "events": history }))
}
