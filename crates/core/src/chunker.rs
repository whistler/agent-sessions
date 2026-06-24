use crate::models::{Chunk, Conversation, Message};
use crate::Result;

pub trait Chunker: Send + Sync {
    fn chunk(&self, conv: &Conversation, messages: &[Message]) -> Result<Vec<Chunk>>;
}

/// Default chunker: one chunk per message, no splitting.
pub struct DefaultChunker;

impl Chunker for DefaultChunker {
    fn chunk(&self, conv: &Conversation, messages: &[Message]) -> Result<Vec<Chunk>> {
        use crate::models::Locator;

        let chunks = messages
            .iter()
            .enumerate()
            .map(|(i, msg)| Chunk {
                locator: Locator {
                    conversation_id: conv.id.clone(),
                    message_index: Some(i),
                    chunk_index: Some(0),
                },
                text: msg.content.clone(),
                role: msg.role.clone(),
                harness: conv.harness.clone(),
                project_path: conv.project_path.clone(),
                model: conv.model.clone(),
                timestamp: msg.timestamp,
            })
            .collect();

        Ok(chunks)
    }
}
