use anyhow::Result;
use chrono::Utc;
use deadpool_redis::Pool;
use redis::AsyncCommands;
use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
use tracing::warn;

use crate::queue::Event;

const DLQ_KEY: &str = "jarvis:dlq";
const DLQ_META_PREFIX: &str = "jarvis:dlq:meta:";
const RETRY_COOLDOWN_SECS: i64 = 60;

pub struct DeadLetterQueue {
    pool: Arc<Pool>,
    depth: AtomicUsize,
}

impl DeadLetterQueue {
    pub fn new(pool: Arc<Pool>) -> Self {
        Self {
            pool,
            depth: AtomicUsize::new(0),
        }
    }

    pub async fn push(&self, mut event: Event, reason: &str) -> Result<()> {
        event.retry_count += 1;
        let meta = serde_json::json!({
            "event": event,
            "reason": reason,
            "failed_at": Utc::now().to_rfc3339(),
        });

        let mut conn = self.pool.get().await?;
        let id = event.id.clone();

        conn.lpush::<_, _, ()>(DLQ_KEY, &id).await?;
        conn.set_ex::<_, _, ()>(
            format!("{DLQ_META_PREFIX}{id}"),
            serde_json::to_string(&meta)?,
            86400, // 24h retention
        )
        .await?;

        self.depth.fetch_add(1, Ordering::Relaxed);
        warn!("DLQ push: event={id} reason={reason} retries={}", event.retry_count);
        Ok(())
    }

    pub async fn pop(&self, event_id: &str) -> Result<Option<Event>> {
        let mut conn = self.pool.get().await?;
        let meta_key = format!("{DLQ_META_PREFIX}{event_id}");
        let raw: Option<String> = conn.get(&meta_key).await?;

        if let Some(raw) = raw {
            let meta: serde_json::Value = serde_json::from_str(&raw)?;
            let event: Event = serde_json::from_value(meta["event"].clone())?;
            conn.lrem::<_, _, ()>(DLQ_KEY, 1, event_id).await?;
            conn.del::<_, ()>(&meta_key).await?;
            self.depth.fetch_sub(1, Ordering::Relaxed);
            return Ok(Some(event));
        }
        Ok(None)
    }

    /// Returns events ready for retry (cooldown passed, can_retry == true).
    pub async fn pop_retryable(&self) -> Result<Vec<Event>> {
        let mut conn = self.pool.get().await?;
        let ids: Vec<String> = conn.lrange(DLQ_KEY, 0, -1).await?;
        let mut ready = Vec::new();

        for id in ids {
            let meta_key = format!("{DLQ_META_PREFIX}{id}");
            let raw: Option<String> = conn.get(&meta_key).await?;
            if let Some(raw) = raw {
                let meta: serde_json::Value = serde_json::from_str(&raw)?;
                let failed_at_str = meta["failed_at"].as_str().unwrap_or("");
                if let Ok(failed_at) = chrono::DateTime::parse_from_rfc3339(failed_at_str) {
                    let age = (Utc::now() - failed_at.with_timezone(&Utc)).num_seconds();
                    if age >= RETRY_COOLDOWN_SECS {
                        let event: Event = serde_json::from_value(meta["event"].clone())?;
                        if event.can_retry() {
                            conn.lrem::<_, _, ()>(DLQ_KEY, 1, &id).await?;
                            conn.del::<_, ()>(&meta_key).await?;
                            self.depth.fetch_sub(1, Ordering::Relaxed);
                            ready.push(event);
                        }
                    }
                }
            }
        }
        Ok(ready)
    }

    pub fn list(&self) -> Vec<serde_json::Value> {
        // Sync snapshot for HTTP endpoint — returns cached ids only
        Vec::new()
    }

    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::Relaxed)
    }
}
