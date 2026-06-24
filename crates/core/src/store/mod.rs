use crate::models::{Conversation, Harness, Locator, Role};
use crate::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod sqlite_vec;

#[cfg(feature = "lancedb")]
pub mod lancedb;

/// Metadata about the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub conversation_count: usize,
    pub chunk_count: usize,
    pub store_path: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
}

/// A chunk with its embedding vector, ready to be stored.
/// Deliberately has NO text field — text is in the conversation transcript on disk.
#[derive(Debug, Clone)]
pub struct ChunkVector {
    pub locator: Locator,
    pub vector: Vec<f32>,
    pub role: Role,
    pub harness: Harness,
    pub project_path: Option<String>,
    pub model: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// A stored chunk retrieved from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub locator: Locator,
    pub score: f32,
    pub role: Role,
    pub harness: Harness,
    pub project_path: Option<String>,
    pub model: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// Filter criteria for listing conversations or searching chunks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Filter {
    pub harness: Option<Harness>,
    pub project_path: Option<String>,
    pub model: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

pub trait VectorStore: Send + Sync {
    fn upsert_conversation(&mut self, c: &Conversation) -> Result<()>;
    fn upsert_chunks(&mut self, conv_id: &str, chunks: &[ChunkVector]) -> Result<()>;
    fn has_conversation(&self, id: &str) -> Result<bool>;
    fn list_conversations(&self, filter: &Filter) -> Result<Vec<Conversation>>;
    fn get_conversation(&self, id: &str) -> Result<Option<Conversation>>;
    fn vector_search(&self, vec: &[f32], limit: usize, filter: &Filter) -> Result<Vec<StoredChunk>>;
    fn meta(&self) -> Result<Meta>;
}
