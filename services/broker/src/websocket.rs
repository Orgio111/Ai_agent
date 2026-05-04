use axum::extract::ws::{Message, WebSocket};
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{broker::EventBroker, queue::Event};

/// Handle a WebSocket connection. Protocol:
/// - Client sends `{"action":"subscribe","topic":"...","subscriber_id":"..."}`
/// - Client sends `{"action":"publish", "event": {...}}`
/// - Server streams events as JSON
pub async fn handle_socket(socket: WebSocket, broker: Arc<EventBroker>) {
    let subscriber_id = Uuid::new_v4().to_string();
    let (mut sender, mut receiver) = socket.split();

    info!("WebSocket connected: {subscriber_id}");

    // Register with wildcard initially; client can refine via subscribe message
    broker.register_subscriber(subscriber_id.clone(), "*".into());
    let mut rx = broker.get_receiver(&subscriber_id).unwrap();

    // Spawn task to forward broker events to WS client
    let sub_id_clone = subscriber_id.clone();
    let mut send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let json = serde_json::to_string(&event).unwrap_or_default();
                    if sender.send(Message::Text(json)).await.is_err() {
                        debug!("WS send failed for {sub_id_clone}");
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    warn!("WS subscriber {sub_id_clone} lagged by {n} messages");
                }
                Err(_) => break,
            }
        }
    });

    // Handle incoming WS messages
    let broker_clone = broker.clone();
    let sub_id_recv = subscriber_id.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    handle_ws_message(&text, &sub_id_recv, &broker_clone).await;
                }
                Message::Close(_) => {
                    info!("WS client {sub_id_recv} closed");
                    break;
                }
                Message::Ping(data) => {
                    // Pong handled by axum automatically
                    debug!("WS ping from {sub_id_recv}: {data:?}");
                }
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }

    info!("WebSocket disconnected: {subscriber_id}");
}

async fn handle_ws_message(text: &str, subscriber_id: &str, broker: &Arc<EventBroker>) {
    let Ok(msg) = serde_json::from_str::<serde_json::Value>(text) else {
        warn!("Invalid WS message from {subscriber_id}: {text}");
        return;
    };

    match msg["action"].as_str() {
        Some("subscribe") => {
            let topic = msg["topic"].as_str().unwrap_or("*").to_string();
            broker.register_subscriber(subscriber_id.to_string(), topic.clone());
            debug!("WS {subscriber_id} subscribed to {topic}");
        }
        Some("publish") => {
            if let Ok(event) = serde_json::from_value::<Event>(msg["event"].clone()) {
                let _ = broker.publish(event).await;
            }
        }
        Some("cancel") => {
            if let Some(id) = msg["event_id"].as_str() {
                broker.cancel_event(id);
            }
        }
        _ => {
            debug!("Unknown WS action from {subscriber_id}: {}", msg["action"]);
        }
    }
}
