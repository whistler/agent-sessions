use crate::Result;
use crate::models::{Conversation, Harness, Locator, Role};
use crate::store::{ChunkVector, Filter, Meta, StoredChunk, VectorStore};
use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{Connection, Row, params};
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed vector store with raw f32 blobs.
pub struct SqliteVecStore {
    conn: Mutex<Connection>,
}

impl SqliteVecStore {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.migrate(Some(path.to_string_lossy().to_string()))?;
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.migrate(None)?;
        Ok(store)
    }

    fn migrate(&self, store_path: Option<String>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                harness TEXT NOT NULL,
                harness_version TEXT,
                project_path TEXT,
                repo_url TEXT,
                git_branch TEXT,
                title TEXT,
                started_at TEXT,
                source_path TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                message_ordinal INTEGER NOT NULL,
                chunk_ordinal INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                harness TEXT NOT NULL,
                role TEXT NOT NULL,
                model TEXT,
                timestamp TEXT,
                project_path TEXT,
                vector BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_conv ON chunks(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_harness ON chunks(harness);
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            ",
        )?;

        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', ?1)",
            params![Utc::now().to_rfc3339()],
        )?;
        if let Some(store_path) = store_path {
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('store_path', ?1)",
                params![store_path],
            )?;
        }

        Ok(())
    }

    fn vec_to_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
        blob.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|value| value * value).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
        }
    }

    fn row_to_conversation(row: &Row<'_>) -> rusqlite::Result<Conversation> {
        let harness: String = row.get("harness")?;
        Ok(Conversation {
            id: row.get("id")?,
            harness: Harness::from_str(&harness),
            harness_version: row.get("harness_version")?,
            project_path: row.get("project_path")?,
            repo_url: row.get("repo_url")?,
            git_branch: row.get("git_branch")?,
            title: row.get("title")?,
            started_at: row
                .get::<_, Option<String>>("started_at")?
                .and_then(|value| DateTime::parse_from_rfc3339(&value).ok())
                .map(|value| value.with_timezone(&Utc)),
            source_path: row.get("source_path")?,
            message_count: row.get::<_, i64>("message_count")? as usize,
        })
    }

    fn timestamp_floor() -> DateTime<Utc> {
        Utc.timestamp_opt(0, 0).single().unwrap_or_else(Utc::now)
    }
}

impl VectorStore for SqliteVecStore {
    fn upsert_conversation(&mut self, conversation: &Conversation) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "
            INSERT INTO conversations
            (id, harness, harness_version, project_path, repo_url, git_branch, title, started_at, source_path, message_count, indexed_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            ON CONFLICT(id) DO UPDATE SET
                harness=excluded.harness,
                harness_version=excluded.harness_version,
                project_path=excluded.project_path,
                repo_url=excluded.repo_url,
                git_branch=excluded.git_branch,
                title=excluded.title,
                started_at=excluded.started_at,
                source_path=excluded.source_path,
                message_count=excluded.message_count,
                indexed_at=excluded.indexed_at
            ",
            params![
                conversation.id,
                conversation.harness.as_str(),
                conversation.harness_version,
                conversation.project_path,
                conversation.repo_url,
                conversation.git_branch,
                conversation.title,
                conversation.started_at.map(|value| value.to_rfc3339()),
                conversation.source_path,
                conversation.message_count as i64,
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    fn upsert_chunks(&mut self, conv_id: &str, chunks: &[ChunkVector]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM chunks WHERE conversation_id = ?1",
            params![conv_id],
        )?;
        let mut stmt = conn.prepare(
            "
            INSERT INTO chunks
            (conversation_id, message_ordinal, chunk_ordinal, source_path, harness, role, model, timestamp, project_path, vector)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ",
        )?;

        for chunk in chunks {
            stmt.execute(params![
                conv_id,
                chunk.locator.message_ordinal as i64,
                chunk.locator.chunk_ordinal as i64,
                chunk.locator.source_path,
                chunk.locator.harness,
                role_to_string(&chunk.role),
                chunk.model,
                chunk.timestamp.to_rfc3339(),
                chunk.project_path,
                Self::vec_to_blob(&chunk.vector),
            ])?;
        }

        Ok(())
    }

    fn has_conversation(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM conversations WHERE id = ?1",
            params![id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn list_conversations(&self, filter: &Filter) -> Result<Vec<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT * FROM conversations")?;
        let rows = stmt.query_map([], Self::row_to_conversation)?;
        let mut conversations = rows.collect::<rusqlite::Result<Vec<_>>>()?;

        conversations.retain(|conversation| conversation_matches_filter(conversation, filter));
        conversations.sort_by(|a, b| {
            let a_started = a.started_at.unwrap_or_else(Self::timestamp_floor);
            let b_started = b.started_at.unwrap_or_else(Self::timestamp_floor);
            b_started.cmp(&a_started).then_with(|| a.id.cmp(&b.id))
        });
        Ok(conversations)
    }

    fn get_conversation(&self, id: &str) -> Result<Option<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT * FROM conversations WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(Self::row_to_conversation(row)?))
        } else {
            Ok(None)
        }
    }

    fn vector_search(
        &self,
        vector: &[f32],
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<StoredChunk>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare("SELECT * FROM chunks")?;
        let rows = stmt.query_map([], |row| {
            Ok(ChunkRow {
                conversation_id: row.get("conversation_id")?,
                message_ordinal: row.get::<_, i64>("message_ordinal")? as usize,
                chunk_ordinal: row.get::<_, i64>("chunk_ordinal")? as usize,
                source_path: row.get("source_path")?,
                harness: row.get("harness")?,
                role: row.get("role")?,
                model: row.get("model")?,
                timestamp: row.get("timestamp")?,
                project_path: row.get("project_path")?,
                vector: row.get("vector")?,
            })
        })?;

        let mut scored = Vec::new();
        for row in rows {
            let row = row?;
            let harness = Harness::from_str(&row.harness);
            let timestamp = row
                .timestamp
                .as_deref()
                .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
                .map(|value| value.with_timezone(&Utc))
                .unwrap_or_else(Utc::now);

            if !chunk_matches_filter(&harness, &row.project_path, &row.model, &timestamp, filter) {
                continue;
            }

            let chunk_vector = Self::blob_to_vec(&row.vector);
            let score = Self::cosine(vector, &chunk_vector);
            let role = match row.role.as_str() {
                "assistant" => Role::Assistant,
                "system" => Role::System,
                _ => Role::User,
            };
            scored.push(StoredChunk {
                locator: Locator {
                    conversation_id: row.conversation_id,
                    message_ordinal: row.message_ordinal,
                    chunk_ordinal: row.chunk_ordinal,
                    source_path: row.source_path,
                    harness: row.harness.clone(),
                },
                score,
                role,
                harness,
                project_path: row.project_path,
                model: row.model,
                timestamp,
            });
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(limit);
        Ok(scored)
    }

    fn meta(&self) -> Result<Meta> {
        let conn = self.conn.lock().unwrap();
        let conversation_count: usize =
            conn.query_row("SELECT COUNT(*) FROM conversations", [], |row| {
                row.get::<_, i64>(0)
            })? as usize;
        let chunk_count: usize = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| {
            row.get::<_, i64>(0)
        })? as usize;
        let store_path = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'store_path'",
                [],
                |row| row.get(0),
            )
            .ok();
        let created_at = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'created_at'",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok()
            .and_then(|value| DateTime::parse_from_rfc3339(&value).ok())
            .map(|value| value.with_timezone(&Utc));

        Ok(Meta {
            conversation_count,
            chunk_count,
            store_path,
            created_at,
        })
    }
}

fn role_to_string(role: &Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    }
}

struct ChunkRow {
    conversation_id: String,
    message_ordinal: usize,
    chunk_ordinal: usize,
    source_path: String,
    harness: String,
    role: String,
    model: Option<String>,
    timestamp: Option<String>,
    project_path: Option<String>,
    vector: Vec<u8>,
}

fn conversation_matches_filter(conversation: &Conversation, filter: &Filter) -> bool {
    if filter
        .harness
        .as_ref()
        .is_some_and(|harness| conversation.harness != *harness)
    {
        return false;
    }

    if filter
        .project_path
        .as_ref()
        .is_some_and(|project_path| conversation.project_path.as_ref() != Some(project_path))
    {
        return false;
    }

    if filter.since.as_ref().is_some_and(|since| {
        conversation
            .started_at
            .is_none_or(|started_at| started_at < *since)
    }) {
        return false;
    }

    if filter.until.as_ref().is_some_and(|until| {
        conversation
            .started_at
            .is_none_or(|started_at| started_at > *until)
    }) {
        return false;
    }

    true
}

fn chunk_matches_filter(
    harness: &Harness,
    project_path: &Option<String>,
    model: &Option<String>,
    timestamp: &DateTime<Utc>,
    filter: &Filter,
) -> bool {
    if filter
        .harness
        .as_ref()
        .is_some_and(|value| value != harness)
    {
        return false;
    }
    if filter
        .project_path
        .as_ref()
        .is_some_and(|value| project_path.as_ref() != Some(value))
    {
        return false;
    }
    if filter
        .model
        .as_ref()
        .is_some_and(|value| model.as_ref() != Some(value))
    {
        return false;
    }
    if filter.since.as_ref().is_some_and(|since| timestamp < since) {
        return false;
    }
    if filter.until.as_ref().is_some_and(|until| timestamp > until) {
        return false;
    }
    true
}
