use super::shared::{default_home_dir, discover_jsonl, parse_jsonl_session, read_message_text};
use crate::Result;
use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Locator, Message};

/// Reads Cursor AI session transcripts.
pub struct CursorConnector {
    sessions_dir: std::path::PathBuf,
}

impl CursorConnector {
    pub fn new(sessions_dir: std::path::PathBuf) -> Self {
        Self { sessions_dir }
    }

    pub fn default_sessions_dir() -> std::path::PathBuf {
        default_home_dir(&[".cursor", "sessions"])
    }
}

impl HarnessConnector for CursorConnector {
    fn id(&self) -> &str {
        "cursor"
    }

    fn is_present(&self) -> bool {
        self.sessions_dir.exists()
    }

    fn discover(&self, since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        discover_jsonl(&self.sessions_dir, self.id(), since)
    }

    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        parse_jsonl_session(r, crate::models::Harness::Cursor)
    }

    fn read(&self, locator: &Locator) -> Result<String> {
        read_message_text(locator)
    }
}
