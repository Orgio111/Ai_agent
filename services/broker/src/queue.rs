use chrono::{DateTime, Utc};
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::sync::Mutex;
use uuid::Uuid;

/// Priority levels — higher number = higher priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl Priority {
    pub fn from_str(s: &str) -> Self {
        match s {
            "low" => Self::Low,
            "high" => Self::High,
            "critical" => Self::Critical,
            _ => Self::Normal,
        }
    }

    pub fn for_agent(agent: &str) -> Self {
        match agent {
            "critic" => Self::Critical,
            "planner" => Self::High,
            "researcher" => Self::Normal,
            "optimizer" => Self::Normal,
            "executor" => Self::Normal,
            _ => Self::Normal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    #[serde(default = "new_uuid")]
    pub id: String,
    pub topic: String,
    pub agent_type: Option<String>,
    pub payload: serde_json::Value,
    pub priority: Option<Priority>,
    pub session_id: Option<String>,
    pub correlation_id: Option<String>,
    #[serde(default = "now")]
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub retry_count: u32,
    pub max_retries: Option<u32>,
    pub ttl_seconds: Option<u64>,
}

fn new_uuid() -> String {
    Uuid::new_v4().to_string()
}

fn now() -> DateTime<Utc> {
    Utc::now()
}

impl Event {
    pub fn effective_priority(&self) -> Priority {
        if let Some(p) = self.priority {
            return p;
        }
        if let Some(ref agent) = self.agent_type {
            return Priority::for_agent(agent);
        }
        Priority::Normal
    }

    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl_seconds {
            let age = (Utc::now() - self.created_at).num_seconds();
            return age > ttl as i64;
        }
        false
    }

    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries.unwrap_or(3)
    }
}

/// Thread-safe priority event queue backed by priority_queue crate.
/// Events with higher priority are dequeued first.
pub struct PriorityEventQueue {
    inner: Mutex<PriorityQueue<String, i32>>,
    events: dashmap::DashMap<String, Event>,
}

impl PriorityEventQueue {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PriorityQueue::new()),
            events: dashmap::DashMap::new(),
        }
    }

    pub fn push(&self, event: Event) {
        let priority = event.effective_priority() as i32;
        let id = event.id.clone();
        self.events.insert(id.clone(), event);
        let mut q = self.inner.lock().expect("queue lock poisoned");
        q.push(id, priority);
    }

    pub fn pop(&self) -> Option<Event> {
        let id = {
            let mut q = self.inner.lock().expect("queue lock poisoned");
            q.pop().map(|(id, _)| id)
        }?;
        self.events.remove(&id).map(|(_, ev)| ev)
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("queue lock poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn remove(&self, id: &str) -> Option<Event> {
        {
            let mut q = self.inner.lock().expect("queue lock poisoned");
            q.remove(id);
        }
        self.events.remove(id).map(|(_, ev)| ev)
    }
}

impl Default for PriorityEventQueue {
    fn default() -> Self {
        Self::new()
    }
}
