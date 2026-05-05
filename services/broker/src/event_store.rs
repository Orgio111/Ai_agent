use anyhow::Result;
use deadpool_redis::Pool;
use redis::AsyncCommands;
use std::sync::Arc;

use crate::queue::Event;

const STORE_KEY: &str = "jarvis:events:log";
const META_PREFIX: &str = "jarvis:events:meta:";

pub struct EventStore {
    pool: Arc<Pool>,
}

impl EventStore {
    pub fn new(pool: Arc<Pool>) -> Self {
        Self { pool }
    }

    pub async fn append(&self, event: &Event) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let serialized = serde_json::to_string(event)?;
        conn.lpush::<_, _, ()>(STORE_KEY, &serialized).await?;
        conn.ltrim::<_, ()>(STORE_KEY, 0, 9999).await?; // keep 10k events
        conn.set_ex::<_, _, ()>(
            format!("{META_PREFIX}{}", event.id),
            &serialized,
            3600,
        )
        .await?;
        Ok(())
    }

    pub async fn get(&self, event_id: &str) -> Result<Option<Event>> {
        let mut conn = self.pool.get().await?;
        let raw: Option<String> = conn.get(format!("{META_PREFIX}{event_id}")).await?;
        Ok(raw.and_then(|s| serde_json::from_str(&s).ok()))
    }

    pub async fn recent(&self, limit: usize) -> Result<Vec<serde_json::Value>> {
        let mut conn = self.pool.get().await?;
        let items: Vec<String> = conn.lrange(STORE_KEY, 0, limit as isize - 1).await?;
        Ok(items
            .into_iter()
            .filter_map(|s| serde_json::from_str(&s).ok())
            .collect())
    }
}
