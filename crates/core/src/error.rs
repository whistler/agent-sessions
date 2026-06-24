use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentSessionsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Conversation not found: {0}")]
    NotFound(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Connector error: {0}")]
    Connector(String),

    #[error("Embedder error: {0}")]
    Embedder(String),

    #[error("Store error: {0}")]
    Store(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AgentSessionsError>;
