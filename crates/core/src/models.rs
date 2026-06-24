use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Harness {
    Claude,
    Codex,
    Cursor,
    OpenCode,
    Custom(String),
}

/// Points to a specific location within a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Locator {
    pub conversation_id: String,
    pub message_index: Option<usize>,
    pub chunk_index: Option<usize>,
}

/// A single conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub harness: Harness,
    pub project_path: Option<String>,
    pub model: Option<String>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub message_count: usize,
    pub title: Option<String>,
}

/// A single message within a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub role: Role,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub index: usize,
}

/// A text chunk derived from a message, used for embedding and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub locator: Locator,
    pub text: String,
    pub role: Role,
    pub harness: Harness,
    pub project_path: Option<String>,
    pub model: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// A search result from vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub locator: Locator,
    pub score: f32,
    pub snippet: String,
    pub conversation: Conversation,
}

/// A search result from regex/grep search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepHit {
    pub locator: Locator,
    pub line: String,
    pub line_number: usize,
    pub conversation: Conversation,
}

/// Summary report from a sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    pub conversations_added: usize,
    pub conversations_skipped: usize,
    pub chunks_indexed: usize,
    pub errors: Vec<String>,
    pub duration_ms: u64,
}

/// A lightweight reference to a conversation for discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRef {
    pub id: String,
    pub harness: Harness,
    pub path: std::path::PathBuf,
    pub modified_at: std::time::SystemTime,
}
