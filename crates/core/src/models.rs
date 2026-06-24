use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Harness {
    Claude,
    Codex,
    Cursor,
    OpenCode,
    Custom(String),
}

impl Harness {
    pub fn as_str(&self) -> &str {
        match self {
            Harness::Claude => "claude",
            Harness::Codex => "codex",
            Harness::Cursor => "cursor",
            Harness::OpenCode => "opencode",
            Harness::Custom(value) => value.as_str(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "claude" => Harness::Claude,
            "codex" => Harness::Codex,
            "cursor" => Harness::Cursor,
            "opencode" => Harness::OpenCode,
            other => Harness::Custom(other.to_string()),
        }
    }
}

impl Serialize for Harness {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Harness {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(Harness::from_str(&value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

/// Stable pointer into a source file. Enough for any connector to lazy-read the text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Locator {
    pub conversation_id: String,
    pub message_ordinal: usize,
    pub chunk_ordinal: usize,
    pub source_path: String,
    pub harness: String,
}

/// One session, harness-agnostic. Text-free - metadata only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub harness: Harness,
    pub harness_version: Option<String>,
    pub project_path: Option<String>,
    pub repo_url: Option<String>,
    pub git_branch: Option<String>,
    pub title: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub source_path: String,
    pub message_count: usize,
}

/// Message metadata only - text is read lazily via connector.read(locator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub locator: Locator,
    pub role: Role,
    pub model: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
}

/// Lightweight reference returned by discover(), before full parse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRef {
    pub id: String,
    pub source_path: std::path::PathBuf,
    pub modified_at: std::time::SystemTime,
}

/// Result of a vector or hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub locator: Locator,
    pub score: f32,
    pub snippet: String,
}

/// Result of a regex search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepHit {
    pub locator: Locator,
    pub snippet: String,
}

/// Summary returned by sync().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    pub conversations_indexed: usize,
    pub chunks_added: usize,
    pub harnesses_synced: Vec<String>,
    pub harness_errors: HashMap<String, String>,
}

/// Progress events emitted by SessionIndex::sync_with_progress().
/// Display logic lives in the CLI; the library only emits events.
#[derive(Debug)]
pub enum SyncEvent {
    /// Harness data directory found; about to index `session_count` sessions.
    HarnessStart { harness: String, session_count: usize },
    /// Harness data directory not present on disk — skipped.
    HarnessSkip { harness: String },
    /// Conversation was already in the index — skipped.
    ConversationSkip { harness: String, id: String },
    /// Conversation was parsed, embedded, and stored.
    ConversationIndexed { harness: String, id: String, chunks: usize },
    /// All sessions for this harness have been processed.
    HarnessDone { harness: String, conversations: usize, chunks: usize },
    /// Non-fatal error for one session (sync continues with remaining sessions).
    Error { harness: String, message: String },
}
