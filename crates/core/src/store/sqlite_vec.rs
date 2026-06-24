use crate::models::Conversation;
use crate::store::{ChunkVector, Filter, Meta, StoredChunk, VectorStore};
use crate::Result;
use rusqlite::Connection;
use std::sync::Mutex;

/// SQLite + sqlite-vec backed vector store.
pub struct SqliteVecStore {
    conn: Mutex<Connection>,
}

impl SqliteVecStore {
    pub fn open(path: &std::path::Path) -> crate::Result<Self> {
        let conn = Connection::open(path)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    pub fn open_in_memory() -> crate::Result<Self> {
        let conn = Connection::open_in_memory()?;
        Ok(Self { conn: Mutex::new(conn) })
    }
}

impl VectorStore for SqliteVecStore {
    fn upsert_conversation(&mut self, _c: &Conversation) -> Result<()> {
        todo!("SqliteVecStore::upsert_conversation")
    }

    fn upsert_chunks(&mut self, _conv_id: &str, _chunks: &[ChunkVector]) -> Result<()> {
        todo!("SqliteVecStore::upsert_chunks")
    }

    fn has_conversation(&self, _id: &str) -> Result<bool> {
        todo!("SqliteVecStore::has_conversation")
    }

    fn list_conversations(&self, _filter: &Filter) -> Result<Vec<Conversation>> {
        todo!("SqliteVecStore::list_conversations")
    }

    fn get_conversation(&self, _id: &str) -> Result<Option<Conversation>> {
        todo!("SqliteVecStore::get_conversation")
    }

    fn vector_search(&self, _vec: &[f32], _limit: usize, _filter: &Filter) -> Result<Vec<StoredChunk>> {
        todo!("SqliteVecStore::vector_search")
    }

    fn meta(&self) -> Result<Meta> {
        todo!("SqliteVecStore::meta")
    }
}
