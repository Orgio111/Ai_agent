use anyhow::{Context, Result};
use dashmap::DashMap;
use deadpool_redis::{Config as RedisConfig, Pool as RedisPool, Runtime};
use futures::StreamExt;
use std::{sync::Arc, time::Duration};
use tokio::sync::{broadcast, Notify};
use tracing::{debug, error, info, warn};

use crate::{
    dlq::DeadLetterQueue,
    event_store::EventStore,
    metrics,
    queue::{Event, PriorityEventQueue},
};

/// Maps subscriber_id → set of topics (glob supported)
type Subscriptions = DashMap<String, Vec<String>>;
/// Maps subscriber_id → broadcast sender
type Channels = DashMap<String, broadcast::Sender<Event>>;
/// Maps event_id → cancellation flag
type Cancellations = DashMap<String, bool>;

pub struct EventBroker {
    queue: Arc<PriorityEventQueue>,
    subscriptions: Arc<Subscriptions>,
    channels: Arc<Channels>,
    cancellations: Arc<Cancellations>,
    pub dlq: Arc<DeadLetterQueue>,
    pub event_store: Arc<EventStore>,
    redis_pool: Arc<RedisPool>,
    notify: Arc<Notify>,
}

impl EventBroker {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let cfg = RedisConfig::from_url(redis_url);
        let redis_pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("create redis pool")?;

        // Verify connection
        {
            let mut conn = redis_pool.get().await.context("redis connect")?;
            redis::cmd("PING")
                .query_async::<()>(&mut conn)
                .await
                .context("redis ping")?;
        }

        info!("Redis connected: {redis_url}");

        let event_store = Arc::new(EventStore::new(Arc::new(redis_pool.clone())));
        let dlq = Arc::new(DeadLetterQueue::new(Arc::new(redis_pool.clone())));

        Ok(Self {
            queue: Arc::new(PriorityEventQueue::new()),
            subscriptions: Arc::new(DashMap::new()),
            channels: Arc::new(DashMap::new()),
            cancellations: Arc::new(DashMap::new()),
            dlq,
            event_store,
            redis_pool: Arc::new(redis_pool),
            notify: Arc::new(Notify::new()),
        })
    }

    /// Publish an event into the priority queue + persist to Redis.
    pub async fn publish(&self, mut event: Event) -> Result<String> {
        if event.id.is_empty() {
            event.id = uuid::Uuid::new_v4().to_string();
        }

        // Persist to event store before queuing (write-ahead)
        self.event_store.append(&event).await?;

        let id = event.id.clone();
        let topic = event.topic.clone();

        metrics::EVENTS_PUBLISHED.with_label_values(&[&topic]).inc();
        metrics::QUEUE_DEPTH
            .with_label_values(&[&topic])
            .set(self.queue.len() as f64 + 1.0);

        self.queue.push(event);
        self.notify.notify_one();

        debug!("Published event {id} topic={topic}");
        Ok(id)
    }

    pub fn register_subscriber(&self, subscriber_id: String, topic: String) {
        let (tx, _) = broadcast::channel(512);
        self.channels.insert(subscriber_id.clone(), tx);
        self.subscriptions
            .entry(subscriber_id.clone())
            .or_default()
            .push(topic.clone());
        info!("Subscriber {subscriber_id} registered for topic={topic}");
    }

    pub fn get_receiver(&self, subscriber_id: &str) -> Option<broadcast::Receiver<Event>> {
        self.channels
            .get(subscriber_id)
            .map(|tx| tx.value().subscribe())
    }

    pub fn cancel_event(&self, event_id: &str) {
        self.cancellations.insert(event_id.to_string(), true);
        self.queue.remove(event_id);
        warn!("Event {event_id} cancelled");
    }

    pub fn dlq_list(&self) -> Vec<serde_json::Value> {
        self.dlq.list()
    }

    pub async fn dlq_retry(&self, event_id: &str) -> Result<()> {
        if let Some(event) = self.dlq.pop(event_id).await? {
            self.publish(event).await?;
        }
        Ok(())
    }

    pub async fn health_status(&self) -> serde_json::Value {
        let redis_ok = async {
            let mut conn = self.redis_pool.get().await.ok()?;
            redis::cmd("PING")
                .query_async::<String>(&mut conn)
                .await
                .ok()
        }
        .await
        .is_some();

        serde_json::json!({
            "status": if redis_ok { "healthy" } else { "degraded" },
            "queue_depth": self.queue.len(),
            "subscribers": self.subscriptions.len(),
            "redis": redis_ok,
            "dlq_depth": self.dlq.depth(),
        })
    }

    pub async fn event_history(&self, limit: usize) -> Vec<serde_json::Value> {
        self.event_store.recent(limit).await.unwrap_or_default()
    }

    /// Main dispatch loop: pops events, fans out to matching subscribers.
    pub async fn run_dispatch_loop(&self) {
        info!("Dispatch loop started");
        loop {
            // Wait for events
            self.notify.notified().await;

            while let Some(event) = self.queue.pop() {
                // Skip cancelled events
                if self.cancellations.contains_key(&event.id) {
                    self.cancellations.remove(&event.id);
                    continue;
                }

                // Skip expired events
                if event.is_expired() {
                    warn!("Event {} expired, dropping", event.id);
                    metrics::EVENTS_EXPIRED
                        .with_label_values(&[&event.topic])
                        .inc();
                    continue;
                }

                let topic = event.topic.clone();
                let event_id = event.id.clone();
                let mut delivered = 0usize;

                for sub_entry in self.subscriptions.iter() {
                    let subscriber_id = sub_entry.key().clone();
                    let topics = sub_entry.value().clone();

                    if topics.iter().any(|t| t == "*" || t == &topic) {
                        if let Some(tx) = self.channels.get(&subscriber_id) {
                            match tx.send(event.clone()) {
                                Ok(_) => {
                                    delivered += 1;
                                }
                                Err(_) => {
                                    debug!("Subscriber {subscriber_id} disconnected, removing");
                                    drop(tx);
                                    self.channels.remove(&subscriber_id);
                                    self.subscriptions.remove(&subscriber_id);
                                }
                            }
                        }
                    }
                }

                debug!("Event {event_id} delivered to {delivered} subscribers");
                metrics::EVENTS_DELIVERED
                    .with_label_values(&[&topic])
                    .inc_by(delivered as f64);

                if delivered == 0 {
                    warn!("Event {event_id} topic={topic}: no subscribers, sending to DLQ");
                    let _ = self.dlq.push(event, "no_subscribers").await;
                    metrics::DLQ_DEPTH.set(self.dlq.depth() as f64);
                }

                metrics::QUEUE_DEPTH
                    .with_label_values(&[&topic])
                    .set(self.queue.len() as f64);
            }
        }
    }

    /// Periodically retry DLQ events that have cooled down.
    pub async fn run_dlq_requeue_loop(&self) {
        info!("DLQ requeue loop started");
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        loop {
            interval.tick().await;
            match self.dlq.pop_retryable().await {
                Ok(events) => {
                    for event in events {
                        if event.can_retry() {
                            info!("DLQ retry: event {}", event.id);
                            let _ = self.publish(event).await;
                        } else {
                            warn!("Event {} exhausted retries, permanently failed", event.id);
                            metrics::EVENTS_PERMANENTLY_FAILED.inc();
                        }
                    }
                }
                Err(e) => error!("DLQ requeue error: {e}"),
            }
        }
    }

    /// Periodic health check; removes stale subscribers.
    pub async fn run_health_check_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            let mut stale: Vec<String> = Vec::new();
            for entry in self.channels.iter() {
                if entry.value().receiver_count() == 0 {
                    stale.push(entry.key().clone());
                }
            }
            for id in stale {
                self.channels.remove(&id);
                self.subscriptions.remove(&id);
                debug!("Removed stale subscriber: {id}");
            }
            metrics::ACTIVE_SUBSCRIBERS.set(self.subscriptions.len() as f64);
        }
    }
}
