pub mod chunker;
pub mod connector;
pub mod connectors;
pub mod embedder;
pub mod error;
pub mod models;
pub mod store;

pub use error::{AgentSessionsError, Result};
pub use models::{
    Chunk, Conversation, ConversationRef, GrepHit, Harness, Locator, Message, Role, SearchHit,
    SyncReport,
};

use serde::{Deserialize, Serialize};

/// Configuration for opening a SessionIndex.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Path to the store directory. Defaults to ~/.agent-sessions.
    pub store_path: Option<std::path::PathBuf>,
    /// Embedding vector dimensions.
    pub embed_dimensions: Option<usize>,
}

/// Info about a registered harness connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessInfo {
    pub id: String,
    pub present: bool,
}

/// Query parameters for listing conversations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ListQuery {
    pub harness: Option<Harness>,
    pub project_path: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Paginated list of conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationPage {
    pub items: Vec<Conversation>,
    pub total: usize,
    pub offset: usize,
}

/// The main entry point. Holds all runtime state.
pub struct SessionIndex {
    store: Box<dyn store::VectorStore>,
    embedder: Box<dyn embedder::Embedder>,
    chunker: Box<dyn chunker::Chunker>,
    connectors: Vec<Box<dyn connector::HarnessConnector>>,
}

impl SessionIndex {
    pub fn open(_config: Config) -> Result<Self> {
        todo!("SessionIndex::open")
    }

    pub fn sync(&mut self) -> Result<SyncReport> {
        todo!("SessionIndex::sync")
    }

    pub fn search(&self, _q: &str) -> Result<Vec<SearchHit>> {
        todo!("SessionIndex::search")
    }

    pub fn grep(&self, _pattern: &str) -> Result<Vec<GrepHit>> {
        todo!("SessionIndex::grep")
    }

    pub fn similar(&self, _q: &str) -> Result<Vec<SearchHit>> {
        todo!("SessionIndex::similar")
    }

    pub fn list_conversations(&self, _query: ListQuery) -> Result<ConversationPage> {
        todo!("SessionIndex::list_conversations")
    }

    pub fn get_conversation(&self, _id: &str) -> Result<Option<Conversation>> {
        todo!("SessionIndex::get_conversation")
    }

    pub fn read(&self, _locator: &Locator) -> Result<String> {
        todo!("SessionIndex::read")
    }

    pub fn harnesses(&self) -> Vec<HarnessInfo> {
        self.connectors
            .iter()
            .map(|c| HarnessInfo {
                id: c.id().to_string(),
                present: c.is_present(),
            })
            .collect()
    }

    pub fn register(&mut self, connector: Box<dyn connector::HarnessConnector>) {
        self.connectors.push(connector);
    }
}
