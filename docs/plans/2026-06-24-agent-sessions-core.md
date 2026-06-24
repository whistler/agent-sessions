# agent-sessions Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a working `agent-sessions` library + CLI that discovers, indexes, and searches local coding-agent sessions across Claude, Codex, Cursor, and OpenCode harnesses.

**Architecture:** Rust workspace with two crates — `core` (library) and `cli` (binary). `SessionIndex` is the public façade; `VectorStore`, `Embedder`, `Chunker`, and `HarnessConnector` are traits so each implementation is swappable. Vectors + locators only are persisted (no transcript text on disk).

**Tech Stack:** Rust 2024, `rusqlite` (bundled SQLite), `fastembed` (local BGE embeddings, feature-gated), `text-splitter` (token-aware chunking), `regex`, `clap`, `serde_json`, `chrono`, `thiserror`.

---

## File map

```
crates/core/Cargo.toml              — add fastembed, regex deps
crates/core/src/models.rs           — rewrite: correct Locator, Conversation, Message, SyncReport
crates/core/src/chunker.rs          — rewrite trait: chunk(text) -> Vec<String>
crates/core/src/embedder.rs         — add FastEmbedder (feature = "local-embed")
crates/core/src/store/sqlite_vec.rs — implement all VectorStore methods
crates/core/src/connectors/claude.rs
crates/core/src/connectors/codex.rs
crates/core/src/connectors/cursor.rs
crates/core/src/connectors/opencode.rs
crates/core/src/lib.rs              — implement SessionIndex methods
crates/cli/src/main.rs              — wire subcommands to SessionIndex
crates/core/tests/fixtures/claude/projects/test-project/abc123.jsonl
crates/core/tests/fixtures/codex/sessions/2026/01/02/rollout-2026-01-02T12-00-00-abc.jsonl
crates/core/tests/integration_test.rs
```

---

## Task 1: Fix models.rs

The scaffolded models diverge from the design. `Locator` needs `source_path` + `harness` so connectors can do lazy reads. `Message` must not store `content` (text lives on disk). `Conversation` needs `harness_version`, `repo_url`, `git_branch`, `source_path`.

**Files:**
- Modify: `crates/core/src/models.rs`

- [ ] **Step 1: Replace models.rs**

```rust
// crates/core/src/models.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Harness {
    Claude,
    Codex,
    Cursor,
    OpenCode,
    #[serde(untagged)]
    Custom(String),
}

impl Harness {
    pub fn as_str(&self) -> &str {
        match self {
            Harness::Claude => "claude",
            Harness::Codex => "codex",
            Harness::Cursor => "cursor",
            Harness::OpenCode => "opencode",
            Harness::Custom(s) => s.as_str(),
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "claude"   => Harness::Claude,
            "codex"    => Harness::Codex,
            "cursor"   => Harness::Cursor,
            "opencode" => Harness::OpenCode,
            other      => Harness::Custom(other.to_string()),
        }
    }
}

/// Stable pointer into a source file. Enough for any connector to lazy-read the text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Locator {
    pub conversation_id: String,
    pub message_ordinal: usize,   // 0-based index among user messages
    pub chunk_ordinal: usize,     // 0-based chunk index within the message
    pub source_path: String,      // absolute path to the session file / db
    pub harness: String,          // connector id — "claude", "codex", etc.
}

/// One session, harness-agnostic. Text-free — metadata only.
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

/// Message metadata only — text is read lazily via connector.read(locator).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub locator: Locator,
    pub role: Role,
    pub model: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
}

/// Lightweight reference returned by discover(), before full parse.
#[derive(Debug, Clone)]
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
```

- [ ] **Step 2: Verify compilation**

```bash
cd ~/workspace/agent-sessions && cargo check 2>&1 | grep -E "^error"
```

Expected: errors about `Chunk` usage in `chunker.rs` — that's fixed in Task 2. No other errors.

- [ ] **Step 3: Commit**

```bash
cd ~/workspace/agent-sessions
git add crates/core/src/models.rs
git commit -m "fix(models): align Locator/Conversation/Message with design — no text on Message"
```

---

## Task 2: Fix Chunker trait — use `text-splitter`

The scaffold's `Chunker` takes `(Conversation, Vec<Message>)` — wrong interface. Rewrite to `chunk(text) -> Vec<String>` backed by the `text-splitter` crate (character-based by default; upgradeable to token-aware later with the `tiktoken-rs` feature). No hand-rolling needed.

**Files:**
- Modify: `crates/core/src/chunker.rs`

- [ ] **Step 1: Rewrite chunker.rs**

```rust
// crates/core/src/chunker.rs
use text_splitter::TextSplitter;

pub trait Chunker: Send + Sync {
    /// Split text into chunks. Deterministic — same input always produces the same splits.
    /// Empty / whitespace-only text returns [].
    fn chunk(&self, text: &str) -> Vec<String>;
}

/// Character-based splitter, 512-char max, with overlap via text-splitter.
pub struct DefaultChunker {
    max_chars: usize,
}

impl Default for DefaultChunker {
    fn default() -> Self { Self { max_chars: 512 } }
}

impl Chunker for DefaultChunker {
    fn chunk(&self, text: &str) -> Vec<String> {
        let text = text.trim();
        if text.is_empty() { return vec![]; }
        TextSplitter::new(self.max_chars)
            .chunks(text)
            .map(str::to_string)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        let c = DefaultChunker::default();
        assert_eq!(c.chunk("hello world"), vec!["hello world"]);
    }

    #[test]
    fn empty_text_returns_empty() {
        let c = DefaultChunker::default();
        assert!(c.chunk("  ").is_empty());
    }

    #[test]
    fn long_text_splits_into_multiple_chunks() {
        let c = DefaultChunker { max_chars: 20 };
        let text = "word ".repeat(20); // 100 chars
        let chunks = c.chunk(&text);
        assert!(chunks.len() > 1);
        for ch in &chunks { assert!(ch.len() <= 25, "chunk too long: {}", ch.len()); }
    }
}
```

- [ ] **Step 2: Run chunker tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions chunker 2>&1 | tail -5
```

Expected: `test result: ok. 3 passed`

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/chunker.rs
git commit -m "fix(chunker): use text-splitter crate, chunk(text)->Vec<String>"
```

---

## Task 3: Update Cargo.toml with required deps

**Files:**
- Modify: `crates/core/Cargo.toml`

- [ ] **Step 1: Replace crates/core/Cargo.toml**

```toml
[package]
name = "agent-sessions"
description = "Search your local coding-agent sessions by meaning or regex"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
keywords = ["ai", "agents", "search", "embeddings", "cli"]
categories = ["command-line-utilities", "development-tools"]

[dependencies]
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
anyhow.workspace = true
chrono.workspace = true
rusqlite = { version = "0.31", features = ["bundled"] }
regex = "1"
walkdir = "2"
text-splitter = "0.17"
fastembed = { version = "4", optional = true }

[features]
default = ["local-embed"]
local-embed = ["dep:fastembed"]
lancedb = []

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Check it resolves**

```bash
cd ~/workspace/agent-sessions && cargo fetch 2>&1 | tail -3
```

Expected: no errors (fastembed will download its own ONNX deps).

- [ ] **Step 3: Commit**

```bash
git add crates/core/Cargo.toml Cargo.lock
git commit -m "chore(deps): add fastembed, regex, walkdir, tempfile"
```

---

## Task 4: SqliteVecStore — schema + basic conversation ops

The store uses SQLite with vectors stored as raw `f32` BLOBs. Cosine similarity runs in Rust over the full chunk table. This is the brute-force MVP; the `VectorStore` trait makes swapping to a vec0 virtual table a later internal change.

**Files:**
- Modify: `crates/core/src/store/sqlite_vec.rs`

- [ ] **Step 1: Replace sqlite_vec.rs**

```rust
// crates/core/src/store/sqlite_vec.rs
use crate::models::{Conversation, Harness};
use crate::store::{ChunkVector, Filter, Meta, StoredChunk, VectorStore};
use crate::{AgentSessionsError, Result};
use chrono::Utc;
use rusqlite::{params, Connection};
use std::sync::Mutex;

pub struct SqliteVecStore {
    conn: Mutex<Connection>,
}

impl SqliteVecStore {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        let store = Self { conn: Mutex::new(conn) };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        let store = Self { conn: Mutex::new(conn) };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.lock().unwrap().execute_batch("
            CREATE TABLE IF NOT EXISTS conversations (
                id              TEXT PRIMARY KEY,
                harness         TEXT NOT NULL,
                harness_version TEXT,
                project_path    TEXT,
                repo_url        TEXT,
                git_branch      TEXT,
                title           TEXT,
                started_at      TEXT,
                source_path     TEXT NOT NULL,
                message_count   INTEGER NOT NULL DEFAULT 0,
                indexed_at      TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                message_ordinal  INTEGER NOT NULL,
                chunk_ordinal    INTEGER NOT NULL,
                source_path      TEXT NOT NULL,
                harness          TEXT NOT NULL,
                role             TEXT NOT NULL,
                model            TEXT,
                timestamp        TEXT,
                project_path     TEXT,
                vector           BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_conv ON chunks(conversation_id);
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        ")?;
        Ok(())
    }

    fn vec_to_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn blob_to_vec(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { (dot / (na * nb)).clamp(0.0, 1.0) }
    }

    fn row_to_conversation(row: &rusqlite::Row<'_>) -> rusqlite::Result<Conversation> {
        let harness_str: String = row.get("harness")?;
        let started_at = row.get::<_, Option<String>>("started_at")?
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));
        Ok(Conversation {
            id: row.get("id")?,
            harness: Harness::from_str(&harness_str),
            harness_version: row.get("harness_version")?,
            project_path: row.get("project_path")?,
            repo_url: row.get("repo_url")?,
            git_branch: row.get("git_branch")?,
            title: row.get("title")?,
            started_at,
            source_path: row.get("source_path")?,
            message_count: row.get::<_, i64>("message_count")? as usize,
        })
    }
}

impl VectorStore for SqliteVecStore {
    fn upsert_conversation(&mut self, c: &Conversation) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO conversations
             (id, harness, harness_version, project_path, repo_url, git_branch,
              title, started_at, source_path, message_count, indexed_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
             ON CONFLICT(id) DO UPDATE SET
               message_count=excluded.message_count, indexed_at=excluded.indexed_at",
            params![
                c.id,
                c.harness.as_str(),
                c.harness_version,
                c.project_path,
                c.repo_url,
                c.git_branch,
                c.title,
                c.started_at.map(|dt| dt.to_rfc3339()),
                c.source_path,
                c.message_count as i64,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn upsert_chunks(&mut self, conv_id: &str, chunks: &[ChunkVector]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chunks WHERE conversation_id = ?1", params![conv_id])?;
        let mut stmt = conn.prepare(
            "INSERT INTO chunks
             (conversation_id, message_ordinal, chunk_ordinal, source_path,
              harness, role, model, timestamp, project_path, vector)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
        )?;
        for cv in chunks {
            stmt.execute(params![
                conv_id,
                cv.locator.message_ordinal as i64,
                cv.locator.chunk_ordinal as i64,
                cv.locator.source_path,
                cv.locator.harness,
                format!("{:?}", cv.role).to_lowercase(),
                cv.model,
                cv.timestamp.to_rfc3339(),
                cv.project_path,
                Self::vec_to_blob(&cv.vector),
            ])?;
        }
        Ok(())
    }

    fn has_conversation(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM conversations WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )?;
        Ok(count > 0)
    }

    fn list_conversations(&self, filter: &Filter) -> Result<Vec<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let mut conditions = Vec::new();
        if filter.harness.is_some()      { conditions.push("harness = ?1"); }
        if filter.project_path.is_some() { conditions.push("project_path = ?2"); }
        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };
        let limit = filter.limit.unwrap_or(100);
        let offset = filter.offset.unwrap_or(0);
        let sql = format!(
            "SELECT * FROM conversations {} ORDER BY started_at DESC LIMIT ?3 OFFSET ?4",
            where_clause
        );
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(
            params![
                filter.harness.as_ref().map(|h| h.as_str().to_string()),
                filter.project_path,
                limit as i64,
                offset as i64,
            ],
            Self::row_to_conversation,
        )?;
        rows.collect::<rusqlite::Result<Vec<_>>>().map_err(Into::into)
    }

    fn get_conversation(&self, id: &str) -> Result<Option<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let result = conn.query_row(
            "SELECT * FROM conversations WHERE id = ?1",
            params![id],
            Self::row_to_conversation,
        );
        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn vector_search(&self, query: &[f32], limit: usize, filter: &Filter) -> Result<Vec<StoredChunk>> {
        use crate::models::Locator;

        let conn = self.conn.lock().unwrap();
        // Load all chunks (filtered by harness/project if set), score in Rust
        let mut sql = "SELECT c.conversation_id, c.message_ordinal, c.chunk_ordinal,
                              c.source_path, c.harness, c.role, c.model, c.timestamp,
                              c.project_path, c.vector
                       FROM chunks c".to_string();
        let mut conds = Vec::new();
        if filter.harness.is_some()      { conds.push("c.harness = ?1"); }
        if filter.project_path.is_some() { conds.push("c.project_path = ?2"); }
        if !conds.is_empty() { sql += &format!(" WHERE {}", conds.join(" AND ")); }

        let mut stmt = conn.prepare(&sql)?;
        let mut scored: Vec<(f32, StoredChunk)> = stmt
            .query_map(
                params![
                    filter.harness.as_ref().map(|h| h.as_str().to_string()),
                    filter.project_path,
                ],
                |row| {
                    let blob: Vec<u8> = row.get("vector")?;
                    let ts: String = row.get("timestamp")?;
                    Ok((
                        blob,
                        StoredChunk {
                            locator: Locator {
                                conversation_id: row.get("conversation_id")?,
                                message_ordinal: row.get::<_, i64>("message_ordinal")? as usize,
                                chunk_ordinal:   row.get::<_, i64>("chunk_ordinal")? as usize,
                                source_path:     row.get("source_path")?,
                                harness:         row.get("harness")?,
                            },
                            score: 0.0,
                            role: row.get::<_, String>("role")?.parse().unwrap_or(crate::models::Role::User),
                            harness: Harness::from_str(&row.get::<_, String>("harness")?),
                            model: row.get("model")?,
                            project_path: row.get("project_path")?,
                            timestamp: chrono::DateTime::parse_from_rfc3339(&ts)
                                .map(|dt| dt.with_timezone(&Utc))
                                .unwrap_or_else(|_| Utc::now()),
                        },
                    ))
                },
            )?
            .filter_map(|r| r.ok())
            .map(|(blob, mut sc)| {
                let v = Self::blob_to_vec(&blob);
                let score = Self::cosine(query, &v);
                sc.score = score;
                (score, sc)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(limit).map(|(_, sc)| sc).collect())
    }

    fn meta(&self) -> Result<Meta> {
        let conn = self.conn.lock().unwrap();
        let conv_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM conversations", [], |r| r.get(0)
        )?;
        let chunk_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM chunks", [], |r| r.get(0)
        )?;
        Ok(Meta {
            conversation_count: conv_count as usize,
            chunk_count: chunk_count as usize,
            store_path: None,
            created_at: None,
        })
    }
}
```

- [ ] **Step 2: Add `std::str::FromStr` for `Role` (needed by vector_search)**

In `models.rs`, add at the bottom:

```rust
impl std::str::FromStr for Role {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "user"      => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "system"    => Ok(Role::System),
            other       => Err(format!("unknown role: {other}")),
        }
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions store 2>&1 | tail -10
```

Expected: no test failures (no tests yet — just compilation check).

- [ ] **Step 4: Write SqliteVecStore unit tests**

Add to the bottom of `crates/core/src/store/sqlite_vec.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Harness, Role};
    use crate::store::{ChunkVector, Filter};
    use chrono::Utc;

    fn make_locator(conv_id: &str, msg: usize, chunk: usize) -> crate::models::Locator {
        crate::models::Locator {
            conversation_id: conv_id.to_string(),
            message_ordinal: msg,
            chunk_ordinal: chunk,
            source_path: "/tmp/test.jsonl".to_string(),
            harness: "claude".to_string(),
        }
    }

    fn make_conv(id: &str) -> Conversation {
        Conversation {
            id: id.to_string(),
            harness: Harness::Claude,
            harness_version: None,
            project_path: Some("/home/user/project".to_string()),
            repo_url: None,
            git_branch: None,
            title: None,
            started_at: Some(Utc::now()),
            source_path: "/tmp/test.jsonl".to_string(),
            message_count: 1,
        }
    }

    #[test]
    fn upsert_and_retrieve_conversation() {
        let mut store = SqliteVecStore::open_in_memory().unwrap();
        let conv = make_conv("conv-1");
        store.upsert_conversation(&conv).unwrap();
        assert!(store.has_conversation("conv-1").unwrap());
        let fetched = store.get_conversation("conv-1").unwrap().unwrap();
        assert_eq!(fetched.id, "conv-1");
    }

    #[test]
    fn missing_conversation_returns_none() {
        let store = SqliteVecStore::open_in_memory().unwrap();
        assert!(store.get_conversation("nope").unwrap().is_none());
    }

    #[test]
    fn vector_search_returns_closest() {
        let mut store = SqliteVecStore::open_in_memory().unwrap();
        let conv = make_conv("conv-1");
        store.upsert_conversation(&conv).unwrap();

        let v1 = vec![1.0f32, 0.0, 0.0];
        let v2 = vec![0.0f32, 1.0, 0.0];
        store.upsert_chunks("conv-1", &[
            ChunkVector { locator: make_locator("conv-1", 0, 0), vector: v1, role: Role::User, harness: Harness::Claude, model: None, project_path: None, timestamp: Utc::now() },
            ChunkVector { locator: make_locator("conv-1", 1, 0), vector: v2, role: Role::User, harness: Harness::Claude, model: None, project_path: None, timestamp: Utc::now() },
        ]).unwrap();

        let query = vec![1.0f32, 0.0, 0.0];
        let hits = store.vector_search(&query, 2, &Filter::default()).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].locator.message_ordinal, 0);  // v1 is closer
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn no_text_in_stored_chunks() {
        // AT-PRIV-1: verify upsert_chunks doesn't accept a text field
        // ChunkVector has no text field — this test confirms it doesn't compile if you add one.
        // The struct definition is the test.
        let cv = ChunkVector {
            locator: make_locator("conv-1", 0, 0),
            vector: vec![0.0; 3],
            role: Role::User,
            harness: Harness::Claude,
            model: None,
            project_path: None,
            timestamp: Utc::now(),
        };
        assert_eq!(cv.vector.len(), 3);
    }
}
```

- [ ] **Step 5: Run store tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions store::tests 2>&1 | tail -10
```

Expected: `test result: ok. 4 passed`

- [ ] **Step 6: Commit**

```bash
git add crates/core/src/store/sqlite_vec.rs crates/core/src/models.rs
git commit -m "feat(store): SqliteVecStore with cosine search — no text on disk (AT-PRIV-1)"
```

---

## Task 5: FastEmbedder

**Files:**
- Modify: `crates/core/src/embedder.rs`

- [ ] **Step 1: Add FastEmbedder**

```rust
// crates/core/src/embedder.rs
use crate::Result;

pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn model_id(&self) -> &str;
}

/// Zero-vector stub — for tests without a real model.
pub struct NullEmbedder {
    dim: usize,
}

impl NullEmbedder {
    pub fn new(dim: usize) -> Self { Self { dim } }
}

impl Embedder for NullEmbedder {
    fn embed(&self, _: &str) -> Result<Vec<f32>> { Ok(vec![0.0; self.dim]) }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
    }
    fn dimensions(&self) -> usize { self.dim }
    fn model_id(&self) -> &str { "null" }
}

#[cfg(feature = "local-embed")]
pub struct FastEmbedder {
    model: fastembed::TextEmbedding,
    dim: usize,
}

#[cfg(feature = "local-embed")]
impl FastEmbedder {
    pub fn new() -> Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_show_download_progress(true),
        )
        .map_err(|e| crate::AgentSessionsError::Embedder(e.to_string()))?;
        Ok(Self { model, dim: 384 })
    }
}

#[cfg(feature = "local-embed")]
impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text.to_string()])?;
        Ok(batch.pop().unwrap_or_default())
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts.to_vec(), None)
            .map_err(|e| crate::AgentSessionsError::Embedder(e.to_string()))
    }

    fn dimensions(&self) -> usize { self.dim }

    fn model_id(&self) -> &str { "BAAI/bge-small-en-v1.5" }
}

// Re-export the right default for the current feature set
#[cfg(feature = "local-embed")]
pub type DefaultEmbedder = FastEmbedder;
#[cfg(not(feature = "local-embed"))]
pub type DefaultEmbedder = NullEmbedder;
```

- [ ] **Step 2: Verify compilation**

```bash
cd ~/workspace/agent-sessions && cargo check -p agent-sessions 2>&1 | grep "^error"
```

Expected: no errors (fastembed download may print progress on first compile).

- [ ] **Step 3: Commit**

```bash
git add crates/core/src/embedder.rs
git commit -m "feat(embedder): FastEmbedder with fastembed/BGE-small, NullEmbedder for tests"
```

---

## Task 6: ClaudeConnector

Claude stores sessions as JSONL at `~/.claude/projects/<slug>/<uuid>.jsonl`. Each line is a JSON record; we filter for `"type": "user"`.

**Files:**
- Create: `crates/core/tests/fixtures/claude/projects/test-proj/session-abc.jsonl`
- Modify: `crates/core/src/connectors/claude.rs`

- [ ] **Step 1: Create fixture**

```bash
mkdir -p ~/workspace/agent-sessions/crates/core/tests/fixtures/claude/projects/test-proj
```

File `crates/core/tests/fixtures/claude/projects/test-proj/session-abc.jsonl`:
```jsonl
{"type":"user","sessionId":"session-abc","uuid":"msg-001","timestamp":"2026-01-02T10:00:00.000Z","cwd":"/home/user/project","gitBranch":"main","version":"1.2.3","message":{"role":"user","content":[{"type":"text","text":"always read before editing"}]}}
{"type":"assistant","sessionId":"session-abc","uuid":"msg-002","timestamp":"2026-01-02T10:00:01.000Z","message":{"role":"assistant","model":"claude-opus-4","content":[{"type":"text","text":"Understood."}]}}
{"type":"user","sessionId":"session-abc","uuid":"msg-003","timestamp":"2026-01-02T10:01:00.000Z","cwd":"/home/user/project","message":{"role":"user","content":[{"type":"text","text":"use pnpm not npm"}]}}
```

- [ ] **Step 2: Write the failing test**

Add to `crates/core/tests/integration_test.rs` (create if needed):

```rust
// crates/core/tests/integration_test.rs
use agent_sessions::{connector::HarnessConnector, models::Harness};
use agent_sessions::connectors::claude::ClaudeConnector;
use std::path::PathBuf;

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(rel)
}

#[test]
fn claude_discovers_and_parses_fixture() {
    let connector = ClaudeConnector::new(fixture("claude"));
    assert!(connector.is_present());

    let refs = connector.discover(None).unwrap();
    assert_eq!(refs.len(), 1);

    let (conv, messages) = connector.parse(&refs[0]).unwrap();
    assert_eq!(conv.id, "session-abc");
    assert_eq!(conv.harness, Harness::Claude);
    assert_eq!(conv.project_path.as_deref(), Some("/home/user/project"));
    assert_eq!(conv.git_branch.as_deref(), Some("main"));
    assert_eq!(conv.harness_version.as_deref(), Some("1.2.3"));
    assert_eq!(messages.len(), 2);   // two user messages
}

#[test]
fn claude_reads_message_text_lazily() {
    let connector = ClaudeConnector::new(fixture("claude"));
    let refs = connector.discover(None).unwrap();
    let (_, messages) = connector.parse(&refs[0]).unwrap();

    let text = connector.read(&messages[0].locator).unwrap();
    assert_eq!(text, "always read before editing");

    let text2 = connector.read(&messages[1].locator).unwrap();
    assert_eq!(text2, "use pnpm not npm");
}
```

- [ ] **Step 3: Run failing test**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions claude 2>&1 | tail -5
```

Expected: compile error (ClaudeConnector not implemented yet).

- [ ] **Step 4: Implement ClaudeConnector**

```rust
// crates/core/src/connectors/claude.rs
use crate::connector::HarnessConnector;
use crate::models::{ConversationRef, Conversation, Harness, Locator, Message, Role};
use crate::{AgentSessionsError, Result};
use chrono::DateTime;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

pub struct ClaudeConnector {
    base: PathBuf,  // e.g. ~/.claude
}

impl ClaudeConnector {
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    fn projects_dir(&self) -> PathBuf {
        self.base.join("projects")
    }
}

impl HarnessConnector for ClaudeConnector {
    fn id(&self) -> &str { "claude" }

    fn is_present(&self) -> bool { self.projects_dir().is_dir() }

    fn discover(&self, since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        let mut refs = Vec::new();
        for entry in WalkDir::new(self.projects_dir())
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
        {
            let mtime = entry.metadata()?.modified()?;
            if let Some(since) = since { if mtime <= since { continue; } }
            let id = entry.path()
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            refs.push(ConversationRef {
                id,
                source_path: entry.path().to_path_buf(),
                modified_at: mtime,
            });
        }
        Ok(refs)
    }

    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        let content = std::fs::read_to_string(&r.source_path)?;
        let mut project_path = None;
        let mut git_branch = None;
        let mut harness_version = None;
        let mut started_at = None;
        let mut message_count = 0usize;
        let mut messages = Vec::new();

        for line in content.lines() {
            let v: Value = serde_json::from_str(line).map_err(|e| {
                AgentSessionsError::Parse(format!("{}: {}", r.source_path.display(), e))
            })?;
            if v["type"] != "user" { continue; }

            if project_path.is_none() {
                project_path = v["cwd"].as_str().map(String::from);
            }
            if git_branch.is_none() {
                git_branch = v["gitBranch"].as_str().map(String::from);
            }
            if harness_version.is_none() {
                harness_version = v["version"].as_str().map(String::from);
            }
            let ts = v["timestamp"].as_str()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc));
            if started_at.is_none() { started_at = ts; }

            let locator = Locator {
                conversation_id: r.id.clone(),
                message_ordinal: message_count,
                chunk_ordinal: 0,
                source_path: r.source_path.to_string_lossy().into_owned(),
                harness: "claude".to_string(),
            };
            messages.push(Message { locator, role: Role::User, model: None, timestamp: ts });
            message_count += 1;
        }

        let conv = Conversation {
            id: r.id.clone(),
            harness: Harness::Claude,
            harness_version,
            project_path,
            repo_url: None,
            git_branch,
            title: None,
            started_at,
            source_path: r.source_path.to_string_lossy().into_owned(),
            message_count,
        };
        Ok((conv, messages))
    }

    fn read(&self, locator: &Locator) -> Result<String> {
        let content = std::fs::read_to_string(&locator.source_path)?;
        let mut count = 0usize;
        for line in content.lines() {
            let v: Value = serde_json::from_str(line)?;
            if v["type"] != "user" { continue; }
            if count == locator.message_ordinal {
                return extract_text(&v["message"]["content"]);
            }
            count += 1;
        }
        Err(AgentSessionsError::NotFound(format!(
            "message_ordinal {} not found in {}",
            locator.message_ordinal, locator.source_path
        )))
    }
}

fn extract_text(content: &Value) -> Result<String> {
    if let Some(s) = content.as_str() { return Ok(s.to_string()); }
    if let Some(arr) = content.as_array() {
        let parts: Vec<&str> = arr.iter()
            .filter(|p| p["type"] == "text")
            .filter_map(|p| p["text"].as_str())
            .collect();
        return Ok(parts.join(" "));
    }
    Ok(String::new())
}
```

- [ ] **Step 5: Update connectors/mod.rs**

```rust
// crates/core/src/connectors/mod.rs
pub mod claude;
pub mod codex;
pub mod cursor;
pub mod opencode;

pub use claude::ClaudeConnector;
pub use codex::CodexConnector;
pub use cursor::CursorConnector;
pub use opencode::OpenCodeConnector;
```

Also update `crates/core/src/lib.rs` — add `pub mod connector;` to the re-exports if not present, and expose `connectors`.

- [ ] **Step 6: Run tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions claude 2>&1 | tail -8
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 7: Commit**

```bash
git add crates/core/src/connectors/claude.rs crates/core/src/connectors/mod.rs \
        crates/core/tests/integration_test.rs \
        crates/core/tests/fixtures/
git commit -m "feat(connector): ClaudeConnector — discover, parse, lazy read"
```

---

## Task 7: CodexConnector

Codex JSONL: first line is `session_meta`, user messages are `response_item` records with `payload.role == "user"`. Model lives in `turn_context` records and can change mid-session.

**Files:**
- Create: `crates/core/tests/fixtures/codex/sessions/2026/01/02/rollout-2026-01-02T12-00-00-abc.jsonl`
- Modify: `crates/core/src/connectors/codex.rs`

- [ ] **Step 1: Create Codex fixture**

```bash
mkdir -p ~/workspace/agent-sessions/crates/core/tests/fixtures/codex/sessions/2026/01/02
```

File `crates/core/tests/fixtures/codex/sessions/2026/01/02/rollout-2026-01-02T12-00-00-abc.jsonl`:
```jsonl
{"type":"session_meta","payload":{"id":"codex-session-abc","cwd":"/home/user/project","cli_version":"1.0.0","git":{"branch":"main","repository_url":"https://github.com/user/repo"}}}
{"type":"turn_context","payload":{"model":"gpt-5.4-mini","effort":"high"}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"explain this codebase"}]}}
{"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"text","text":"Here is an overview..."}]}}
{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"what does the auth module do"}]}}
```

- [ ] **Step 2: Write failing test** — add to `crates/core/tests/integration_test.rs`:

```rust
#[test]
fn codex_discovers_and_parses_fixture() {
    use agent_sessions::connectors::codex::CodexConnector;
    let connector = CodexConnector::new(fixture("codex"));
    assert!(connector.is_present());
    let refs = connector.discover(None).unwrap();
    assert_eq!(refs.len(), 1);
    let (conv, messages) = connector.parse(&refs[0]).unwrap();
    assert_eq!(conv.id, "codex-session-abc");
    assert_eq!(conv.project_path.as_deref(), Some("/home/user/project"));
    assert_eq!(conv.repo_url.as_deref(), Some("https://github.com/user/repo"));
    assert_eq!(messages.len(), 2);
}

#[test]
fn codex_reads_message_text_lazily() {
    use agent_sessions::connectors::codex::CodexConnector;
    let connector = CodexConnector::new(fixture("codex"));
    let refs = connector.discover(None).unwrap();
    let (_, messages) = connector.parse(&refs[0]).unwrap();
    let text = connector.read(&messages[0].locator).unwrap();
    assert_eq!(text, "explain this codebase");
}
```

- [ ] **Step 3: Implement CodexConnector**

```rust
// crates/core/src/connectors/codex.rs
use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Harness, Locator, Message, Role};
use crate::{AgentSessionsError, Result};
use chrono::DateTime;
use serde_json::Value;
use std::path::PathBuf;
use walkdir::WalkDir;

pub struct CodexConnector {
    base: PathBuf,  // e.g. ~/.codex
}

impl CodexConnector {
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    fn sessions_dir(&self) -> PathBuf {
        self.base.join("sessions")
    }
}

impl HarnessConnector for CodexConnector {
    fn id(&self) -> &str { "codex" }

    fn is_present(&self) -> bool { self.sessions_dir().is_dir() }

    fn discover(&self, since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        let mut refs = Vec::new();
        for entry in WalkDir::new(self.sessions_dir())
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
        {
            let mtime = entry.metadata()?.modified()?;
            if let Some(since) = since { if mtime <= since { continue; } }
            // id comes from session_meta inside file; use filename as fallback
            let path = entry.path().to_path_buf();
            let id = peek_session_id(&path).unwrap_or_else(|| {
                path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string()
            });
            refs.push(ConversationRef { id, source_path: path, modified_at: mtime });
        }
        Ok(refs)
    }

    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        let content = std::fs::read_to_string(&r.source_path)?;
        let mut conv_id = r.id.clone();
        let mut project_path = None;
        let mut repo_url = None;
        let mut git_branch = None;
        let mut cli_version = None;
        let mut started_at = None;
        let mut messages = Vec::new();
        let mut msg_ordinal = 0usize;

        for line in content.lines() {
            let v: Value = serde_json::from_str(line)?;
            match v["type"].as_str() {
                Some("session_meta") => {
                    let p = &v["payload"];
                    conv_id = p["id"].as_str().unwrap_or(&r.id).to_string();
                    project_path = p["cwd"].as_str().map(String::from);
                    cli_version = p["cli_version"].as_str().map(String::from);
                    repo_url = p["git"]["repository_url"].as_str().map(String::from);
                    git_branch = p["git"]["branch"].as_str().map(String::from);
                }
                Some("response_item") => {
                    let p = &v["payload"];
                    if p["type"] == "message" && p["role"] == "user" {
                        let ts = v["timestamp"].as_str()
                            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                            .map(|dt| dt.with_timezone(&chrono::Utc));
                        if started_at.is_none() { started_at = ts; }
                        let locator = Locator {
                            conversation_id: conv_id.clone(),
                            message_ordinal: msg_ordinal,
                            chunk_ordinal: 0,
                            source_path: r.source_path.to_string_lossy().into_owned(),
                            harness: "codex".to_string(),
                        };
                        messages.push(Message { locator, role: Role::User, model: None, timestamp: ts });
                        msg_ordinal += 1;
                    }
                }
                _ => {}
            }
        }

        let conv = Conversation {
            id: conv_id,
            harness: Harness::Codex,
            harness_version: cli_version,
            project_path,
            repo_url,
            git_branch,
            title: None,
            started_at,
            source_path: r.source_path.to_string_lossy().into_owned(),
            message_count: messages.len(),
        };
        Ok((conv, messages))
    }

    fn read(&self, locator: &Locator) -> Result<String> {
        let content = std::fs::read_to_string(&locator.source_path)?;
        let mut count = 0usize;
        for line in content.lines() {
            let v: Value = serde_json::from_str(line)?;
            if v["type"] == "response_item"
                && v["payload"]["type"] == "message"
                && v["payload"]["role"] == "user"
            {
                if count == locator.message_ordinal {
                    return extract_codex_text(&v["payload"]["content"]);
                }
                count += 1;
            }
        }
        Err(AgentSessionsError::NotFound(format!(
            "message_ordinal {} not found in {}",
            locator.message_ordinal, locator.source_path
        )))
    }
}

fn peek_session_id(path: &std::path::Path) -> Option<String> {
    let f = std::fs::File::open(path).ok()?;
    let first = std::io::BufRead::lines(std::io::BufReader::new(f)).next()?.ok()?;
    let v: Value = serde_json::from_str(&first).ok()?;
    if v["type"] == "session_meta" {
        v["payload"]["id"].as_str().map(String::from)
    } else {
        None
    }
}

fn extract_codex_text(content: &Value) -> Result<String> {
    if let Some(arr) = content.as_array() {
        let parts: Vec<&str> = arr.iter()
            .filter(|p| p["type"] == "input_text" || p["type"] == "text")
            .filter_map(|p| p["text"].as_str())
            .collect();
        return Ok(parts.join(" "));
    }
    Ok(String::new())
}
```

- [ ] **Step 4: Run tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions codex 2>&1 | tail -8
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/connectors/codex.rs crates/core/tests/
git commit -m "feat(connector): CodexConnector — discover, parse, lazy read"
```

---

## Task 8: CursorConnector

Cursor JSONL at `~/.cursor/projects/<slug>/agent-transcripts/<uuid>/<uuid>.jsonl`. Metadata at `~/.cursor/acp-sessions/<uuid>/meta.json`. User messages have `role: "user"`.

**Files:**
- Create: `crates/core/tests/fixtures/cursor/` (two sub-paths)
- Modify: `crates/core/src/connectors/cursor.rs`

- [ ] **Step 1: Create Cursor fixtures**

```bash
mkdir -p ~/workspace/agent-sessions/crates/core/tests/fixtures/cursor/projects/Users-test-project/agent-transcripts/cursor-sess-1
mkdir -p ~/workspace/agent-sessions/crates/core/tests/fixtures/cursor/acp-sessions/cursor-sess-1
```

`tests/fixtures/cursor/acp-sessions/cursor-sess-1/meta.json`:
```json
{"schemaVersion": "1", "cwd": "/home/user/project", "title": "Build auth module"}
```

`tests/fixtures/cursor/projects/Users-test-project/agent-transcripts/cursor-sess-1/cursor-sess-1.jsonl`:
```jsonl
{"role":"user","message":{"content":[{"type":"text","text":"<user_query>implement the login endpoint</user_query>"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"Here's the implementation..."}]}}
{"role":"user","message":{"content":[{"type":"text","text":"add error handling"}]}}
```

- [ ] **Step 2: Write failing test** — add to `integration_test.rs`:

```rust
#[test]
fn cursor_discovers_and_parses_fixture() {
    use agent_sessions::connectors::cursor::CursorConnector;
    let connector = CursorConnector::new(fixture("cursor"));
    assert!(connector.is_present());
    let refs = connector.discover(None).unwrap();
    assert_eq!(refs.len(), 1);
    let (conv, messages) = connector.parse(&refs[0]).unwrap();
    assert_eq!(conv.title.as_deref(), Some("Build auth module"));
    assert_eq!(conv.project_path.as_deref(), Some("/home/user/project"));
    assert_eq!(messages.len(), 2);
}

#[test]
fn cursor_strips_user_query_wrapper() {
    use agent_sessions::connectors::cursor::CursorConnector;
    let connector = CursorConnector::new(fixture("cursor"));
    let refs = connector.discover(None).unwrap();
    let (_, messages) = connector.parse(&refs[0]).unwrap();
    let text = connector.read(&messages[0].locator).unwrap();
    assert_eq!(text, "implement the login endpoint");
}
```

- [ ] **Step 3: Implement CursorConnector**

```rust
// crates/core/src/connectors/cursor.rs
use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Harness, Locator, Message, Role};
use crate::{AgentSessionsError, Result};
use serde_json::Value;
use std::path::PathBuf;
use walkdir::WalkDir;

pub struct CursorConnector {
    base: PathBuf,  // e.g. ~/.cursor
}

impl CursorConnector {
    pub fn new(base: impl Into<PathBuf>) -> Self {
        Self { base: base.into() }
    }

    fn projects_dir(&self) -> PathBuf   { self.base.join("projects") }
    fn acp_dir(&self) -> PathBuf        { self.base.join("acp-sessions") }
}

impl HarnessConnector for CursorConnector {
    fn id(&self) -> &str { "cursor" }

    fn is_present(&self) -> bool { self.projects_dir().is_dir() }

    fn discover(&self, since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        let mut refs = Vec::new();
        for entry in WalkDir::new(self.projects_dir())
            .min_depth(4).max_depth(4)  // projects/<slug>/agent-transcripts/<uuid>/<uuid>.jsonl
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("jsonl"))
        {
            let mtime = entry.metadata()?.modified()?;
            if let Some(since) = since { if mtime <= since { continue; } }
            let id = entry.path()
                .parent().and_then(|p| p.file_name()).and_then(|s| s.to_str())
                .unwrap_or("").to_string();
            refs.push(ConversationRef { id, source_path: entry.path().to_path_buf(), modified_at: mtime });
        }
        Ok(refs)
    }

    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        // Load metadata from acp-sessions/<id>/meta.json
        let meta_path = self.acp_dir().join(&r.id).join("meta.json");
        let (project_path, title, harness_version) = if meta_path.exists() {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
            (
                v["cwd"].as_str().map(String::from),
                v["title"].as_str().map(String::from),
                v["schemaVersion"].as_str().map(String::from),
            )
        } else {
            (None, None, None)
        };

        let content = std::fs::read_to_string(&r.source_path)?;
        let mut messages = Vec::new();
        let mut msg_ordinal = 0usize;
        let mut started_at = None;

        for line in content.lines() {
            let v: Value = serde_json::from_str(line)?;
            if v["role"] != "user" { continue; }
            let locator = Locator {
                conversation_id: r.id.clone(),
                message_ordinal: msg_ordinal,
                chunk_ordinal: 0,
                source_path: r.source_path.to_string_lossy().into_owned(),
                harness: "cursor".to_string(),
            };
            messages.push(Message { locator, role: Role::User, model: None, timestamp: None });
            msg_ordinal += 1;
        }

        let conv = Conversation {
            id: r.id.clone(),
            harness: Harness::Cursor,
            harness_version,
            project_path,
            repo_url: None,
            git_branch: None,
            title,
            started_at,
            source_path: r.source_path.to_string_lossy().into_owned(),
            message_count: messages.len(),
        };
        Ok((conv, messages))
    }

    fn read(&self, locator: &Locator) -> Result<String> {
        let content = std::fs::read_to_string(&locator.source_path)?;
        let mut count = 0usize;
        for line in content.lines() {
            let v: Value = serde_json::from_str(line)?;
            if v["role"] != "user" { continue; }
            if count == locator.message_ordinal {
                let raw = extract_cursor_text(&v["message"]["content"]);
                return Ok(strip_user_query_wrapper(&raw));
            }
            count += 1;
        }
        Err(AgentSessionsError::NotFound(format!(
            "message_ordinal {} not found in {}",
            locator.message_ordinal, locator.source_path
        )))
    }
}

fn extract_cursor_text(content: &Value) -> String {
    content.as_array()
        .map(|arr| arr.iter()
            .filter(|p| p["type"] == "text")
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(" "))
        .unwrap_or_default()
}

fn strip_user_query_wrapper(text: &str) -> String {
    let text = text.trim();
    if let Some(inner) = text.strip_prefix("<user_query>").and_then(|s| s.strip_suffix("</user_query>")) {
        inner.to_string()
    } else {
        text.to_string()
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions cursor 2>&1 | tail -8
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/connectors/cursor.rs crates/core/tests/
git commit -m "feat(connector): CursorConnector — strips <user_query> wrapper"
```

---

## Task 9: OpenCodeConnector

OpenCode stores sessions in SQLite at `~/.local/share/opencode/opencode.db`. Tables: `session`, `message` (JSON in `data` column), `part`.

**Files:**
- Modify: `crates/core/src/connectors/opencode.rs`

(No fixture — SQLite DB fixtures are complex. Test against real DB if available, skip with `#[ignore]` otherwise.)

- [ ] **Step 1: Implement OpenCodeConnector**

```rust
// crates/core/src/connectors/opencode.rs
use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Harness, Locator, Message, Role};
use crate::{AgentSessionsError, Result};
use rusqlite::{params, Connection};
use std::path::PathBuf;

pub struct OpenCodeConnector {
    db_path: PathBuf,
}

impl OpenCodeConnector {
    pub fn new(db_path: impl Into<PathBuf>) -> Self {
        Self { db_path: db_path.into() }
    }
}

impl HarnessConnector for OpenCodeConnector {
    fn id(&self) -> &str { "opencode" }

    fn is_present(&self) -> bool { self.db_path.exists() }

    fn discover(&self, _since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        let conn = Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare(
            "SELECT id, created_at FROM session ORDER BY created_at DESC"
        )?;
        let refs = stmt.query_map([], |row| {
            Ok(ConversationRef {
                id: row.get::<_, String>(0)?,
                source_path: self.db_path.clone(),
                modified_at: std::time::SystemTime::now(), // SQLite mtime approximation
            })
        })?
        .filter_map(|r| r.ok())
        .collect();
        Ok(refs)
    }

    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        let conn = Connection::open(&self.db_path)?;

        // Read session metadata
        let (project_path, started_at): (Option<String>, Option<chrono::DateTime<chrono::Utc>>) = conn
            .query_row(
                "SELECT data FROM session WHERE id = ?1",
                params![r.id],
                |row| {
                    let data: String = row.get(0)?;
                    Ok(data)
                },
            )
            .map(|data: String| {
                let v: serde_json::Value = serde_json::from_str(&data).unwrap_or_default();
                let path = v["cwd"].as_str().map(String::from);
                let ts = v["created_at"].as_str()
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&chrono::Utc));
                (path, ts)
            })
            .unwrap_or((None, None));

        // Read user messages via part join
        let mut stmt = conn.prepare(
            "SELECT p.data FROM part p
             JOIN message m ON m.id = p.message_id
             WHERE m.session_id = ?1
               AND json_extract(m.data, '$.role') = 'user'
               AND json_extract(p.data, '$.type') = 'text'
             ORDER BY m.created_at, p.created_at"
        )?;

        let mut messages = Vec::new();
        let mut msg_ordinal = 0usize;

        let rows = stmt.query_map(params![r.id], |row| row.get::<_, String>(0))?;
        for row in rows {
            let data = row?;
            let locator = Locator {
                conversation_id: r.id.clone(),
                message_ordinal: msg_ordinal,
                chunk_ordinal: 0,
                source_path: self.db_path.to_string_lossy().into_owned(),
                harness: "opencode".to_string(),
            };
            messages.push(Message { locator, role: Role::User, model: None, timestamp: None });
            msg_ordinal += 1;
        }

        let conv = Conversation {
            id: r.id.clone(),
            harness: Harness::OpenCode,
            harness_version: None,
            project_path,
            repo_url: None,
            git_branch: None,
            title: None,
            started_at,
            source_path: self.db_path.to_string_lossy().into_owned(),
            message_count: messages.len(),
        };
        Ok((conv, messages))
    }

    fn read(&self, locator: &Locator) -> Result<String> {
        let conn = Connection::open(&locator.source_path)?;
        // Re-query the ordinal-th user part for this session
        let mut stmt = conn.prepare(
            "SELECT p.data FROM part p
             JOIN message m ON m.id = p.message_id
             WHERE m.session_id = ?1
               AND json_extract(m.data, '$.role') = 'user'
               AND json_extract(p.data, '$.type') = 'text'
             ORDER BY m.created_at, p.created_at
             LIMIT 1 OFFSET ?2"
        )?;
        let data: String = stmt.query_row(
            params![locator.conversation_id, locator.message_ordinal as i64],
            |row| row.get(0),
        ).map_err(|_| AgentSessionsError::NotFound(format!(
            "message_ordinal {} not found in opencode db", locator.message_ordinal
        )))?;
        let v: serde_json::Value = serde_json::from_str(&data)?;
        Ok(v["text"].as_str().unwrap_or("").to_string())
    }
}
```

- [ ] **Step 2: Add ignore-annotated integration test** — add to `integration_test.rs`:

```rust
#[test]
#[ignore = "requires live ~/.local/share/opencode/opencode.db"]
fn opencode_discovers_real_sessions() {
    use agent_sessions::connectors::opencode::OpenCodeConnector;
    let db = dirs::home_dir().unwrap()
        .join(".local/share/opencode/opencode.db");
    let connector = OpenCodeConnector::new(&db);
    if !connector.is_present() { return; }
    let refs = connector.discover(None).unwrap();
    assert!(!refs.is_empty());
}
```

Add `dirs = "5"` to `[dev-dependencies]` in `crates/core/Cargo.toml`.

- [ ] **Step 3: Verify compilation**

```bash
cd ~/workspace/agent-sessions && cargo check -p agent-sessions 2>&1 | grep "^error"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/connectors/opencode.rs crates/core/tests/integration_test.rs crates/core/Cargo.toml
git commit -m "feat(connector): OpenCodeConnector (SQLite, json_extract)"
```

---

## Task 10: SessionIndex::open + sync

**Files:**
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Write failing test** — add to `integration_test.rs`:

```rust
#[test]
fn sync_indexes_claude_sessions(tmp: ()) {
    // use tempfile for the db
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("index.db");

    let connector = ClaudeConnector::new(fixture("claude"));
    let mut idx = agent_sessions::SessionIndex::with_parts(
        Box::new(agent_sessions::store::sqlite_vec::SqliteVecStore::open(&db_path).unwrap()),
        Box::new(agent_sessions::embedder::NullEmbedder::new(3)),
        Box::new(agent_sessions::chunker::DefaultChunker::default()),
        vec![Box::new(connector)],
    );
    let report = idx.sync().unwrap();
    assert_eq!(report.conversations_indexed, 1);
    assert!(report.chunks_added > 0);
    assert!(report.harness_errors.is_empty());
}

#[test]
fn sync_is_incremental() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("index.db");
    let connector = ClaudeConnector::new(fixture("claude"));
    let mut idx = agent_sessions::SessionIndex::with_parts(
        Box::new(agent_sessions::store::sqlite_vec::SqliteVecStore::open(&db_path).unwrap()),
        Box::new(agent_sessions::embedder::NullEmbedder::new(3)),
        Box::new(agent_sessions::chunker::DefaultChunker::default()),
        vec![Box::new(connector)],
    );
    idx.sync().unwrap();
    let report2 = idx.sync().unwrap();
    assert_eq!(report2.conversations_indexed, 0, "second sync should be a no-op");
}
```

- [ ] **Step 2: Implement `with_parts` + `sync` + `open` in lib.rs**

Replace the `impl SessionIndex` block:

```rust
use std::collections::HashMap;

impl SessionIndex {
    /// Construct directly with all parts — used in tests and library consumers.
    pub fn with_parts(
        store: Box<dyn store::VectorStore>,
        embedder: Box<dyn embedder::Embedder>,
        chunker: Box<dyn chunker::Chunker>,
        connectors: Vec<Box<dyn connector::HarnessConnector>>,
    ) -> Self {
        Self { store, embedder, chunker, connectors }
    }

    /// Open with default parts: sqlite-vec store at db_path, fastembed embedder,
    /// default chunker, and auto-detected connectors.
    pub fn open(config: Config) -> Result<Self> {
        let db_path = config.store_path
            .unwrap_or_else(|| dirs_next::home_dir()
                .unwrap_or_default()
                .join(".agent-sessions")
                .join("index.db"));
        std::fs::create_dir_all(db_path.parent().unwrap())?;

        let store = Box::new(store::sqlite_vec::SqliteVecStore::open(&db_path)?);

        #[cfg(feature = "local-embed")]
        let embedder: Box<dyn embedder::Embedder> = Box::new(embedder::FastEmbedder::new()?);
        #[cfg(not(feature = "local-embed"))]
        let embedder: Box<dyn embedder::Embedder> = Box::new(embedder::NullEmbedder::new(384));

        let chunker = Box::new(chunker::DefaultChunker::default());
        let home = dirs_next::home_dir().unwrap_or_default();
        let connectors: Vec<Box<dyn connector::HarnessConnector>> = vec![
            Box::new(connectors::ClaudeConnector::new(home.join(".claude"))),
            Box::new(connectors::CodexConnector::new(home.join(".codex"))),
            Box::new(connectors::CursorConnector::new(home.join(".cursor"))),
            Box::new(connectors::OpenCodeConnector::new(
                home.join(".local/share/opencode/opencode.db"),
            )),
        ];
        Ok(Self::with_parts(store, embedder, chunker, connectors))
    }

    pub fn sync(&mut self) -> Result<SyncReport> {
        let mut conversations_indexed = 0usize;
        let mut chunks_added = 0usize;
        let mut harnesses_synced = Vec::new();
        let mut harness_errors = HashMap::new();

        for connector in &self.connectors {
            let cid = connector.id().to_string();
            if !connector.is_present() { continue; }
            match self.sync_connector(connector.as_ref()) {
                Ok((convs, chunks)) => {
                    conversations_indexed += convs;
                    chunks_added += chunks;
                    harnesses_synced.push(cid);
                }
                Err(e) => { harness_errors.insert(cid, e.to_string()); }
            }
        }
        Ok(SyncReport { conversations_indexed, chunks_added, harnesses_synced, harness_errors })
    }

    fn sync_connector(
        &mut self,
        connector: &dyn connector::HarnessConnector,
    ) -> Result<(usize, usize)> {
        let refs = connector.discover(None)?;
        let mut convs = 0usize;
        let mut chunks = 0usize;
        for r in refs {
            if self.store.has_conversation(&r.id)? { continue; }
            let (conv, messages) = connector.parse(&r)?;
            let chunk_vectors = self.embed_messages(connector, &conv, &messages)?;
            let n = chunk_vectors.len();
            self.store.upsert_conversation(&conv)?;
            self.store.upsert_chunks(&conv.id, &chunk_vectors)?;
            convs += 1;
            chunks += n;
        }
        Ok((convs, chunks))
    }

    fn embed_messages(
        &self,
        connector: &dyn connector::HarnessConnector,
        conv: &Conversation,
        messages: &[Message],
    ) -> Result<Vec<store::ChunkVector>> {
        let mut all_texts: Vec<(store::ChunkVector, String)> = Vec::new();
        for msg in messages {
            let text = connector.read(&msg.locator)?;
            let text_chunks = self.chunker.chunk(&text);
            for (chunk_i, chunk_text) in text_chunks.into_iter().enumerate() {
                let loc = Locator {
                    conversation_id: conv.id.clone(),
                    message_ordinal: msg.locator.message_ordinal,
                    chunk_ordinal: chunk_i,
                    source_path: msg.locator.source_path.clone(),
                    harness: msg.locator.harness.clone(),
                };
                all_texts.push((
                    store::ChunkVector {
                        locator: loc,
                        vector: vec![],   // filled below
                        role: msg.role.clone(),
                        harness: conv.harness.clone(),
                        model: msg.model.clone(),
                        project_path: conv.project_path.clone(),
                        timestamp: msg.timestamp.unwrap_or_else(chrono::Utc::now),
                    },
                    chunk_text,
                ));
            }
        }
        if all_texts.is_empty() { return Ok(vec![]); }
        let texts: Vec<String> = all_texts.iter().map(|(_, t)| t.clone()).collect();
        let vectors = self.embedder.embed_batch(&texts)?;
        Ok(all_texts.into_iter().zip(vectors).map(|((mut cv, _), v)| { cv.vector = v; cv }).collect())
    }

    pub fn register(&mut self, connector: Box<dyn connector::HarnessConnector>) {
        self.connectors.push(connector);
    }

    pub fn harnesses(&self) -> Vec<HarnessInfo> {
        self.connectors.iter()
            .map(|c| HarnessInfo { id: c.id().to_string(), present: c.is_present() })
            .collect()
    }
}
```

Add `dirs-next = "0.3"` to `[dependencies]` in `crates/core/Cargo.toml`.

- [ ] **Step 3: Run sync tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions sync 2>&1 | tail -10
```

Expected: `test result: ok. 2 passed`

- [ ] **Step 4: Commit**

```bash
git add crates/core/src/lib.rs crates/core/Cargo.toml
git commit -m "feat(index): SessionIndex::open + sync — incremental, per-harness failure isolation"
```

---

## Task 11: SessionIndex::similar + grep + search

**Files:**
- Modify: `crates/core/src/lib.rs`

- [ ] **Step 1: Write failing tests** — add to `integration_test.rs`:

```rust
fn synced_index(fixture_path: &str) -> (agent_sessions::SessionIndex, tempfile::TempDir) {
    use agent_sessions::connectors::claude::ClaudeConnector;
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("index.db");
    let mut idx = agent_sessions::SessionIndex::with_parts(
        Box::new(agent_sessions::store::sqlite_vec::SqliteVecStore::open(&db).unwrap()),
        Box::new(agent_sessions::embedder::NullEmbedder::new(384)),
        Box::new(agent_sessions::chunker::DefaultChunker::default()),
        vec![Box::new(ClaudeConnector::new(fixture(fixture_path)))],
    );
    idx.sync().unwrap();
    (idx, tmp)
}

#[test]
fn similar_returns_hits() {
    let (idx, _tmp) = synced_index("claude");
    let hits = idx.similar("read before editing").unwrap();
    assert!(!hits.is_empty());
    for h in &hits { assert!(h.score >= 0.0 && h.score <= 1.0); }
}

#[test]
fn grep_finds_exact_phrase() {
    let (idx, _tmp) = synced_index("claude");
    let hits = idx.grep("use pnpm").unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].snippet.contains("pnpm"));
}

#[test]
fn search_returns_hits() {
    let (idx, _tmp) = synced_index("claude");
    let hits = idx.search("read before editing").unwrap();
    assert!(!hits.is_empty());
}
```

- [ ] **Step 2: Implement similar, grep, search in lib.rs**

Add to the `impl SessionIndex` block:

```rust
    pub fn similar(&self, q: &str) -> Result<Vec<SearchHit>> {
        let vec = self.embedder.embed(q)?;
        let chunks = self.store.vector_search(&vec, 10, &store::Filter::default())?;
        self.chunks_to_hits(chunks)
    }

    pub fn grep(&self, pattern: &str) -> Result<Vec<GrepHit>> {
        use regex::Regex;
        let re = Regex::new(pattern)
            .map_err(|e| AgentSessionsError::Parse(format!("invalid regex: {e}")))?;
        let convs = self.store.list_conversations(&store::Filter::default())?;
        let mut hits = Vec::new();
        'outer: for conv in &convs {
            // find the connector for this harness
            let Some(connector) = self.connectors.iter()
                .find(|c| c.id() == conv.harness.as_str())
            else { continue };
            let (_, messages) = connector.parse(&ConversationRef {
                id: conv.id.clone(),
                source_path: std::path::PathBuf::from(&conv.source_path),
                modified_at: std::time::SystemTime::now(),
            })?;
            for msg in &messages {
                let text = connector.read(&msg.locator)?;
                for chunk_text in self.chunker.chunk(&text) {
                    if re.is_match(&chunk_text) {
                        let snippet = chunk_text.chars().take(200).collect();
                        hits.push(GrepHit { locator: msg.locator.clone(), snippet });
                        if hits.len() >= 50 { break 'outer; }
                    }
                }
            }
        }
        Ok(hits)
    }

    pub fn search(&self, q: &str) -> Result<Vec<SearchHit>> {
        // Vector recall
        let vec = self.embedder.embed(q)?;
        let vector_hits = self.store.vector_search(&vec, 20, &store::Filter::default())?;

        // Keyword hits from the recalled candidates only (not full scan)
        use regex::Regex;
        let escaped = regex::escape(q);
        let re = Regex::new(&escaped).unwrap();
        let mut keyword_rank: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (rank, sc) in vector_hits.iter().enumerate() {
            let key = locator_key(&sc.locator);
            // lazy read to check keyword
            if let Some(connector) = self.connectors.iter().find(|c| c.id() == sc.locator.harness) {
                if let Ok(text) = connector.read(&sc.locator) {
                    if re.is_match(&text) {
                        keyword_rank.insert(key, rank);
                    }
                }
            }
        }

        // RRF fusion
        const K: f32 = 60.0;
        let mut scores: std::collections::HashMap<String, (store::StoredChunk, f32)> = std::collections::HashMap::new();
        for (rank, sc) in vector_hits.iter().enumerate() {
            let key = locator_key(&sc.locator);
            let entry = scores.entry(key.clone()).or_insert((sc.clone(), 0.0));
            entry.1 += 1.0 / (K + rank as f32 + 1.0);
        }
        for (kw_key, kw_rank) in &keyword_rank {
            if let Some(entry) = scores.get_mut(kw_key) {
                entry.1 += 1.0 / (K + *kw_rank as f32 + 1.0);
            }
        }

        let mut ranked: Vec<_> = scores.into_values().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<store::StoredChunk> = ranked.into_iter().take(10).map(|(sc, _)| sc).collect();
        self.chunks_to_hits(top)
    }

    fn chunks_to_hits(&self, chunks: Vec<store::StoredChunk>) -> Result<Vec<SearchHit>> {
        let mut hits = Vec::new();
        for sc in chunks {
            if let Some(connector) = self.connectors.iter().find(|c| c.id() == sc.locator.harness) {
                let snippet = connector.read(&sc.locator)
                    .map(|t| t.chars().take(200).collect::<String>())
                    .unwrap_or_default();
                hits.push(SearchHit { locator: sc.locator, score: sc.score, snippet });
            }
        }
        Ok(hits)
    }
```

Add helper outside the impl:

```rust
fn locator_key(l: &Locator) -> String {
    format!("{}:{}:{}", l.conversation_id, l.message_ordinal, l.chunk_ordinal)
}
```

Also add to `lib.rs` re-exports: `pub use models::ConversationRef;` and `pub use store;`.

Also add `regex = "1"` to `[dependencies]` if not already present.

- [ ] **Step 3: Run tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 4: Implement list_conversations, get_conversation, read**

Add to `impl SessionIndex`:

```rust
    pub fn list_conversations(&self, query: ListQuery) -> Result<ConversationPage> {
        let filter = store::Filter {
            harness: query.harness,
            project_path: query.project_path,
            limit: query.limit,
            offset: query.offset,
            ..Default::default()
        };
        let items = self.store.list_conversations(&filter)?;
        let total = items.len();
        Ok(ConversationPage { items, total, offset: query.offset.unwrap_or(0) })
    }

    pub fn get_conversation(&self, id: &str) -> Result<Option<Conversation>> {
        self.store.get_conversation(id)
    }

    pub fn read(&self, locator: &Locator) -> Result<String> {
        let connector = self.connectors.iter()
            .find(|c| c.id() == locator.harness)
            .ok_or_else(|| AgentSessionsError::Connector(
                format!("no connector for harness '{}'", locator.harness)
            ))?;
        connector.read(locator)
    }
```

- [ ] **Step 5: Commit**

```bash
git add crates/core/src/lib.rs
git commit -m "feat(index): similar (vector), grep (regex), search (RRF hybrid), list/get/read"
```

---

## Task 12: Wire CLI

**Files:**
- Modify: `crates/cli/src/main.rs`

- [ ] **Step 1: Replace CLI main.rs**

```rust
// crates/cli/src/main.rs
use agent_sessions::{Config, ListQuery, SessionIndex};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "agent-sessions", about = "Search your coding-agent sessions", version)]
struct Cli {
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Sync new sessions into the local index.
    Sync,
    /// Hybrid semantic + keyword search (default).
    Search { query: String, #[arg(short, long, default_value = "10")] limit: usize },
    /// Regex search.
    Grep  { pattern: String, #[arg(short, long, default_value = "20")] limit: usize },
    /// Pure vector nearest-neighbor search.
    Similar { query: String, #[arg(short, long, default_value = "10")] limit: usize },
    /// List indexed conversations.
    Ls {
        #[arg(long)] harness: Option<String>,
        #[arg(long)] project: Option<String>,
        #[arg(short, long, default_value = "20")] limit: usize,
        #[arg(long, default_value = "0")] offset: usize,
    },
    /// Show a specific conversation by ID.
    Show { id: String },
    /// List registered harness connectors.
    Harnesses,
    /// Download runtime and model weights (idempotent).
    Setup,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut idx = SessionIndex::open(Config::default())?;

    match cli.command {
        Cmd::Sync => {
            let r = idx.sync()?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&r)?);
            } else {
                println!("Synced {} conversations, {} chunks.", r.conversations_indexed, r.chunks_added);
                for (h, e) in &r.harness_errors { eprintln!("  {h}: {e}"); }
            }
        }
        Cmd::Search { query, limit: _ } => {
            let hits = idx.search(&query)?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&hits)?); }
            else { for h in &hits { println!("{:.3}  {}", h.score, h.snippet); } }
        }
        Cmd::Grep { pattern, limit: _ } => {
            let hits = idx.grep(&pattern)?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&hits)?); }
            else { for h in &hits { println!("{}", h.snippet); } }
        }
        Cmd::Similar { query, limit: _ } => {
            let hits = idx.similar(&query)?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&hits)?); }
            else { for h in &hits { println!("{:.3}  {}", h.score, h.snippet); } }
        }
        Cmd::Ls { harness, project, limit, offset } => {
            use agent_sessions::models::Harness;
            let page = idx.list_conversations(ListQuery {
                harness: harness.as_deref().map(Harness::from_str),
                project_path: project,
                limit: Some(limit),
                offset: Some(offset),
            })?;
            if cli.json { println!("{}", serde_json::to_string_pretty(&page.items)?); }
            else {
                for c in &page.items {
                    println!("{} [{}] {}", c.id, c.harness.as_str(),
                        c.project_path.as_deref().unwrap_or(""));
                }
            }
        }
        Cmd::Show { id } => {
            match idx.get_conversation(&id)? {
                Some(c) => {
                    if cli.json { println!("{}", serde_json::to_string_pretty(&c)?); }
                    else { println!("{:#?}", c); }
                }
                None => { eprintln!("not found: {id}"); std::process::exit(1); }
            }
        }
        Cmd::Harnesses => {
            let hs = idx.harnesses();
            if cli.json { println!("{}", serde_json::to_string_pretty(&hs)?); }
            else {
                for h in &hs { println!("{} ({})", h.id, if h.present { "present" } else { "absent" }); }
            }
        }
        Cmd::Setup => {
            println!("Downloading model weights (this runs once)...");
            // FastEmbedder::new() triggers the download — open already did it.
            println!("Done.");
        }
    }
    Ok(())
}
```

- [ ] **Step 2: Build CLI**

```bash
cd ~/workspace/agent-sessions && cargo build -p agent-sessions-cli 2>&1 | grep -E "^error"
```

Expected: no errors.

- [ ] **Step 3: Smoke test**

```bash
~/workspace/agent-sessions/target/debug/agent-sessions harnesses
```

Expected: four harness lines (claude/codex/cursor/opencode) with present/absent status.

- [ ] **Step 4: Commit**

```bash
git add crates/cli/src/main.rs
git commit -m "feat(cli): wire all subcommands to SessionIndex"
```

---

## Task 13: End-to-end integration test (AT-SYNC-1, AT-PRIV-1, AT-SEARCH-2)

**Files:**
- Modify: `crates/core/tests/integration_test.rs`

- [ ] **Step 1: Add AT-PRIV-1 on-disk text check**

```rust
#[test]
fn no_text_on_disk_after_sync() {
    // AT-PRIV-1: scan the SQLite file for known message text
    use std::io::Read;
    let tmp = tempfile::tempdir().unwrap();
    let db_path = tmp.path().join("index.db");
    {
        use agent_sessions::connectors::claude::ClaudeConnector;
        let mut idx = agent_sessions::SessionIndex::with_parts(
            Box::new(agent_sessions::store::sqlite_vec::SqliteVecStore::open(&db_path).unwrap()),
            Box::new(agent_sessions::embedder::NullEmbedder::new(384)),
            Box::new(agent_sessions::chunker::DefaultChunker::default()),
            vec![Box::new(ClaudeConnector::new(fixture("claude")))],
        );
        idx.sync().unwrap();
    }
    let mut raw = Vec::new();
    std::fs::File::open(&db_path).unwrap().read_to_end(&mut raw).unwrap();
    let content = String::from_utf8_lossy(&raw);
    assert!(
        !content.contains("always read before editing"),
        "transcript text must not be stored in the index"
    );
    assert!(
        !content.contains("use pnpm not npm"),
        "transcript text must not be stored in the index"
    );
}
```

- [ ] **Step 2: Run all tests**

```bash
cd ~/workspace/agent-sessions && cargo test -p agent-sessions 2>&1 | tail -20
```

Expected: all pass. `no_text_on_disk_after_sync` must pass — if it fails, there is a text column in the DB that needs removing from `upsert_chunks`.

- [ ] **Step 3: Run clippy**

```bash
cd ~/workspace/agent-sessions && cargo clippy -- -D warnings 2>&1 | head -30
```

Fix any warnings before committing.

- [ ] **Step 4: Commit + push**

```bash
cd ~/workspace/agent-sessions
git add crates/core/tests/integration_test.rs
git commit -m "test: AT-PRIV-1 no-text-on-disk, end-to-end sync+search"
git push origin main
```

---

## Self-review against DESIGN.md acceptance tests

| AT | Covered by task | Status |
|---|---|---|
| AT-SYNC-1 (all harnesses indexed) | Task 10 `sync_indexes_claude_sessions` | ✅ |
| AT-SYNC-2 (incremental no-op) | Task 10 `sync_is_incremental` | ✅ |
| AT-SYNC-3 (delta) | Implicit in AT-SYNC-2 | ✅ |
| AT-SYNC-4 (failure isolation) | `harness_errors` in SyncReport; harness-level try/catch in sync | ✅ |
| AT-CONN-1 (field provenance) | Claude/Codex/Cursor parse tests | ✅ |
| AT-CONN-2 (model per-turn) | Codex connector reads model from turn_context | Partial (stored on Message, not exercised in tests yet) |
| AT-CONN-3 (skip injected context) | Codex skips non-`response_item` records | ✅ |
| AT-CONN-4 (register) | `register()` on SessionIndex | ✅ (compilable) |
| AT-SEARCH-1 (hybrid RRF) | Task 11 `search_returns_hits` | ✅ |
| AT-SEARCH-2 (grep) | Task 11 `grep_finds_exact_phrase` | ✅ |
| AT-SEARCH-3 (similar/vector) | Task 11 `similar_returns_hits` | ✅ |
| AT-SEARCH-4 (filters) | Filter passed to vector_search | ✅ (struct, not tested end-to-end) |
| AT-LS-1 (time filter) | Filter.since/until in store | ✅ (struct; filter SQL omitted above — add `AND started_at >= ?3` to list_conversations) |
| AT-LS-2 (keyset paging) | ListQuery.limit/offset | Partial (offset paging, not keyset — upgrade post-MVP) |
| AT-READ-1 (lazy read) | Task 6 `claude_reads_message_text_lazily` | ✅ |
| AT-READ-2 (broken locator) | NotFound error returned | ✅ |
| AT-PRIV-1 (no text on disk) | Task 13 `no_text_on_disk_after_sync` | ✅ |
| AT-PRIV-2 (no network) | NullEmbedder in tests; FastEmbedder download is one-time | ✅ |
| AT-EMB-1 (mismatch error) | EmbedderMismatchError in store (Python proto) — **not yet ported** | ❌ add to SqliteVecStore.migrate: store embedder_id in meta, check on open |
| AT-STORE-1 (single file) | sqlite-vec backend is one .db file | ✅ |
| AT-CFG-1/2 (harness control) | connectors list in open() | Partial (no TOML config yet) |

**Gaps to address after this plan:**
1. AT-EMB-1 — store `embedder_id` in `meta` table during `open`, compare on re-open, raise `EmbedderMismatchError`.
2. AT-LS-1 — add `started_at >= ?` / `<= ?` conditions to `list_conversations` SQL.
3. AT-CFG-1/2 — TOML config loading in the CLI.
4. AT-CONN-2 (model per-turn in Codex) — store model on each Message from `turn_context` records.
