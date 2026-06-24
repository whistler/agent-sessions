pub mod chunker;
pub mod connector;
pub mod connectors;
pub mod embedder;
pub mod error;
pub mod models;
pub mod store;

pub use chunker::{Chunker, DefaultChunker};
pub use connector::HarnessConnector;
pub use embedder::{Embedder, FastEmbedder, HashEmbedder, default_embedder};
pub use error::{AgentSessionsError, Result};
pub use models::{
    Conversation, ConversationRef, GrepHit, Harness, Locator, Message, Role, SearchHit, SyncReport,
};
pub use store::{ChunkVector, Filter, Meta, StoredChunk, VectorStore};

use crate::connectors::{ClaudeConnector, CodexConnector, CursorConnector, OpenCodeConnector};
use crate::store::sqlite_vec::SqliteVecStore;
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

/// Configuration for opening a SessionIndex.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Path to the store database file. Defaults to ~/.agent-sessions/index.db.
    pub store_path: Option<PathBuf>,
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
    store: Box<dyn VectorStore>,
    embedder: Box<dyn Embedder>,
    chunker: Box<dyn Chunker>,
    connectors: Vec<Box<dyn HarnessConnector>>,
}

impl SessionIndex {
    pub fn open(config: Config) -> Result<Self> {
        let store_path = resolve_store_path(config.store_path);
        let store = Box::new(SqliteVecStore::open(&store_path)?);
        let embedder = default_embedder()?;
        let chunker = Box::new(DefaultChunker::default());
        let connectors = default_connectors();

        Ok(Self {
            store,
            embedder,
            chunker,
            connectors,
        })
    }

    pub fn from_parts(
        store: Box<dyn VectorStore>,
        embedder: Box<dyn Embedder>,
        chunker: Box<dyn Chunker>,
        connectors: Vec<Box<dyn HarnessConnector>>,
    ) -> Self {
        Self {
            store,
            embedder,
            chunker,
            connectors,
        }
    }

    pub fn sync(&mut self) -> Result<SyncReport> {
        let mut report = SyncReport {
            conversations_indexed: 0,
            chunks_added: 0,
            harnesses_synced: Vec::new(),
            harness_errors: std::collections::HashMap::new(),
        };

        for connector in &self.connectors {
            let discovered = match connector.discover(None) {
                Ok(conversations) => conversations,
                Err(err) => {
                    report
                        .harness_errors
                        .insert(connector.id().to_string(), err.to_string());
                    continue;
                }
            };

            let mut harness_indexed = false;
            for conv_ref in discovered {
                let (conversation, messages) = match connector.parse(&conv_ref) {
                    Ok(parsed) => parsed,
                    Err(err) => {
                        report
                            .harness_errors
                            .insert(connector.id().to_string(), err.to_string());
                        continue;
                    }
                };

                let mut chunk_records = Vec::new();
                let mut chunk_texts = Vec::new();
                for message in messages {
                    let locator = message.locator.clone();
                    let role = message.role.clone();
                    let model = message.model.clone();
                    let timestamp = message
                        .timestamp
                        .clone()
                        .or(conversation.started_at.clone())
                        .unwrap_or_else(utc_now);

                    let message_text = match connector.read(&locator) {
                        Ok(text) => text,
                        Err(err) => {
                            report
                                .harness_errors
                                .insert(connector.id().to_string(), err.to_string());
                            continue;
                        }
                    };

                    let chunks = self.chunker.chunk(&message_text);

                    for (chunk_ordinal, chunk_text) in chunks.into_iter().enumerate() {
                        let mut chunk_locator = locator.clone();
                        chunk_locator.chunk_ordinal = chunk_ordinal;
                        chunk_texts.push(chunk_text);
                        chunk_records.push(ChunkVector {
                            locator: chunk_locator,
                            vector: Vec::new(),
                            role: role.clone(),
                            harness: conversation.harness.clone(),
                            project_path: conversation.project_path.clone(),
                            model: model.clone(),
                            timestamp,
                        });
                    }
                }

                let chunk_refs: Vec<&str> = chunk_texts.iter().map(String::as_str).collect();
                let vectors = if chunk_refs.is_empty() {
                    Vec::new()
                } else {
                    self.embedder.embed_batch(&chunk_refs)?
                };

                for (record, vector) in chunk_records.iter_mut().zip(vectors.into_iter()) {
                    record.vector = vector;
                }

                self.store.upsert_conversation(&conversation)?;
                self.store.upsert_chunks(&conversation.id, &chunk_records)?;

                report.conversations_indexed += 1;
                report.chunks_added += chunk_records.len();
                harness_indexed = true;
            }

            if harness_indexed {
                report.harnesses_synced.push(connector.id().to_string());
            }
        }

        Ok(report)
    }

    pub fn search(&self, q: &str) -> Result<Vec<SearchHit>> {
        let filter = Filter::default();
        let query_vector = self.embedder.embed(q)?;
        let candidate_limit = q.split_whitespace().count().saturating_mul(8).max(20);
        let candidates = self
            .store
            .vector_search(&query_vector, candidate_limit, &filter)?;
        let query_terms = token_terms(q);

        let mut scored = Vec::new();
        for (rank, candidate) in candidates.iter().enumerate() {
            let snippet = self.chunk_snippet(&candidate.locator)?;
            let keyword_score = keyword_score(&snippet, &query_terms, q);
            scored.push((candidate, rank, keyword_score, snippet));
        }

        let mut keyword_sorted: Vec<_> = scored
            .iter()
            .enumerate()
            .map(|(rank, (candidate, _, keyword_score, _))| (rank, candidate, *keyword_score))
            .collect();
        keyword_sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut keyword_ranks = std::collections::HashMap::new();
        for (rank, (idx, _, _)) in keyword_sorted.into_iter().enumerate() {
            keyword_ranks.insert(idx, rank);
        }

        let mut hits: Vec<SearchHit> = scored
            .into_iter()
            .enumerate()
            .map(|(idx, (candidate, vector_rank, _, snippet))| {
                let keyword_rank = keyword_ranks
                    .get(&idx)
                    .copied()
                    .unwrap_or(vector_limit_rank(vector_rank));
                let fused = reciprocal_rank_fusion(vector_rank, keyword_rank);
                SearchHit {
                    locator: candidate.locator.clone(),
                    score: fused,
                    snippet,
                }
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(q.split_whitespace().count().max(1).min(20));
        Ok(hits)
    }

    pub fn grep(&self, pattern: &str) -> Result<Vec<GrepHit>> {
        let regex =
            Regex::new(pattern).map_err(|err| AgentSessionsError::Parse(err.to_string()))?;
        let conversations = self.store.list_conversations(&Filter::default())?;
        let mut hits = Vec::new();

        for conversation in conversations {
            let connector = self
                .connectors
                .iter()
                .find(|connector| connector.id() == conversation.harness.as_str())
                .ok_or_else(|| {
                    AgentSessionsError::Connector(conversation.harness.as_str().to_string())
                })?;

            for message_ordinal in 0..conversation.message_count {
                let locator = Locator {
                    conversation_id: conversation.id.clone(),
                    message_ordinal,
                    chunk_ordinal: 0,
                    source_path: conversation.source_path.clone(),
                    harness: conversation.harness.as_str().to_string(),
                };
                let text = connector.read(&locator)?;
                if let Some(snippet) = regex_snippet(&text, &regex) {
                    hits.push(GrepHit { locator, snippet });
                    if hits.len() >= 20 {
                        return Ok(hits);
                    }
                }
            }
        }

        Ok(hits)
    }

    pub fn similar(&self, q: &str) -> Result<Vec<SearchHit>> {
        let filter = Filter::default();
        let query_vector = self.embedder.embed(q)?;
        let candidates = self.store.vector_search(&query_vector, 20, &filter)?;

        let mut hits = Vec::new();
        for candidate in candidates {
            let locator = candidate.locator.clone();
            hits.push(SearchHit {
                locator: locator.clone(),
                score: candidate.score,
                snippet: self.chunk_snippet(&locator)?,
            });
        }

        Ok(hits)
    }

    pub fn list_conversations(&self, query: ListQuery) -> Result<ConversationPage> {
        let filter = Filter {
            harness: query.harness,
            project_path: query.project_path,
            model: None,
            since: None,
            until: None,
            limit: query.limit,
            offset: query.offset,
        };
        let conversations = self.store.list_conversations(&filter)?;
        let offset = query.offset.unwrap_or(0);
        let limit = query.limit.unwrap_or(20);
        let total = conversations.len();
        let items = conversations.into_iter().skip(offset).take(limit).collect();

        Ok(ConversationPage {
            items,
            total,
            offset,
        })
    }

    pub fn get_conversation(&self, id: &str) -> Result<Option<Conversation>> {
        self.store.get_conversation(id)
    }

    pub fn read(&self, locator: &Locator) -> Result<String> {
        let connector = self
            .connectors
            .iter()
            .find(|connector| connector.id() == locator.harness)
            .ok_or_else(|| AgentSessionsError::Connector(locator.harness.clone()))?;
        connector.read(locator)
    }

    pub fn harnesses(&self) -> Vec<HarnessInfo> {
        self.connectors
            .iter()
            .map(|connector| HarnessInfo {
                id: connector.id().to_string(),
                present: connector.is_present(),
            })
            .collect()
    }

    pub fn register(&mut self, connector: Box<dyn HarnessConnector>) {
        self.connectors.push(connector);
    }

    fn chunk_snippet(&self, locator: &Locator) -> Result<String> {
        let text = self.read(locator)?;
        let chunks = self.chunker.chunk(&text);
        if let Some(chunk) = chunks.get(locator.chunk_ordinal) {
            Ok(chunk.clone())
        } else {
            Ok(text)
        }
    }
}

fn default_connectors() -> Vec<Box<dyn HarnessConnector>> {
    vec![
        Box::new(ClaudeConnector::new(ClaudeConnector::default_sessions_dir())),
        Box::new(CodexConnector::new(CodexConnector::default_sessions_dir())),
        Box::new(CursorConnector::new(CursorConnector::default_sessions_dir())),
        Box::new(OpenCodeConnector::new(
            OpenCodeConnector::default_sessions_dir(),
        )),
    ]
}

fn resolve_store_path(path: Option<PathBuf>) -> PathBuf {
    match path {
        Some(path) if path.extension().is_some() => path,
        Some(path) => path.join("index.db"),
        None => default_home_dir(&[".agent-sessions", "index.db"]),
    }
}

fn default_home_dir(subdir: &[&str]) -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    subdir.iter().fold(home, |path, segment| path.join(segment))
}

fn utc_now() -> DateTime<Utc> {
    DateTime::<Utc>::from(SystemTime::now())
}

fn token_terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

fn keyword_score(text: &str, terms: &[String], query: &str) -> f32 {
    let lowered = text.to_lowercase();
    let mut score = 0.0;
    if lowered.contains(&query.to_lowercase()) {
        score += 4.0;
    }
    score
        + terms
            .iter()
            .filter(|term| lowered.contains(term.as_str()))
            .count() as f32
}

fn reciprocal_rank_fusion(vector_rank: usize, keyword_rank: usize) -> f32 {
    let k = 60.0;
    1.0 / (k + vector_rank as f32) + 1.0 / (k + keyword_rank as f32)
}

fn vector_limit_rank(rank: usize) -> usize {
    rank + 1000
}

fn regex_snippet(text: &str, regex: &Regex) -> Option<String> {
    for line in text.lines() {
        if regex.is_match(line) {
            return Some(line.trim().to_string());
        }
    }

    if regex.is_match(text) {
        return Some(text.lines().next().unwrap_or(text).trim().to_string());
    }

    None
}
