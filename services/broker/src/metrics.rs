use once_cell::sync::Lazy;
use prometheus::{register_counter_vec, register_gauge, register_gauge_vec, CounterVec, Gauge, GaugeVec};

pub static EVENTS_PUBLISHED: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("jarvis_broker_events_published_total", "Events published", &["topic"]).unwrap()
});

pub static EVENTS_DELIVERED: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("jarvis_broker_events_delivered_total", "Events delivered to subscribers", &["topic"]).unwrap()
});

pub static EVENTS_EXPIRED: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!("jarvis_broker_events_expired_total", "Events expired before delivery", &["topic"]).unwrap()
});

pub static EVENTS_PERMANENTLY_FAILED: Lazy<prometheus::Counter> = Lazy::new(|| {
    prometheus::register_counter!("jarvis_broker_events_permanently_failed_total", "Events that exhausted all retries").unwrap()
});

pub static QUEUE_DEPTH: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec!("jarvis_broker_queue_depth", "Current queue depth by topic", &["topic"]).unwrap()
});

pub static DLQ_DEPTH: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!("jarvis_broker_dlq_depth", "Dead-letter queue depth").unwrap()
});

pub static ACTIVE_SUBSCRIBERS: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!("jarvis_broker_active_subscribers", "Active WebSocket subscribers").unwrap()
});
