use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Locator, Message};
use crate::Result;

/// Reads Cursor AI session transcripts.
pub struct CursorConnector {
    sessions_dir: std::path::PathBuf,
}

impl CursorConnector {
    pub fn new(sessions_dir: std::path::PathBuf) -> Self {
        Self { sessions_dir }
    }
}

impl HarnessConnector for CursorConnector {
    fn id(&self) -> &str {
        "cursor"
    }

    fn is_present(&self) -> bool {
        self.sessions_dir.exists()
    }

    fn discover(&self, _since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        todo!("CursorConnector::discover")
    }

    fn parse(&self, _r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        todo!("CursorConnector::parse")
    }

    fn read(&self, _locator: &Locator) -> Result<String> {
        todo!("CursorConnector::read")
    }
}
