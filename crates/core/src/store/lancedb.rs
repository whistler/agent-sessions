//! LanceDB-backed vector store. Enabled with the `lancedb` cargo feature.

use crate::models::Conversation;
use crate::store::{ChunkVector, Filter, Meta, StoredChunk, VectorStore};
use crate::Result;

/// LanceDB backed vector store (feature-gated).
pub struct LanceDbStore {
    _path: std::path::PathBuf,
}

impl LanceDbStore {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        Ok(Self {
            _path: path.to_path_buf(),
        })
    }
}

impl VectorStore for LanceDbStore {
    fn upsert_conversation(&mut self, _c: &Conversation) -> Result<()> {
        todo!("LanceDbStore::upsert_conversation")
    }

    fn upsert_chunks(&mut self, _conv_id: &str, _chunks: &[ChunkVector]) -> Result<()> {
        todo!("LanceDbStore::upsert_chunks")
    }

    fn has_conversation(&self, _id: &str) -> Result<bool> {
        todo!("LanceDbStore::has_conversation")
    }

    fn list_conversations(&self, _filter: &Filter) -> Result<Vec<Conversation>> {
        todo!("LanceDbStore::list_conversations")
    }

    fn get_conversation(&self, _id: &str) -> Result<Option<Conversation>> {
        todo!("LanceDbStore::get_conversation")
    }

    fn vector_search(&self, _vec: &[f32], _limit: usize, _filter: &Filter) -> Result<Vec<StoredChunk>> {
        todo!("LanceDbStore::vector_search")
    }

    fn meta(&self) -> Result<Meta> {
        todo!("LanceDbStore::meta")
    }
}
