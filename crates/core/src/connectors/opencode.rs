use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Locator, Message};
use crate::Result;

/// Reads opencode session transcripts.
pub struct OpenCodeConnector {
    sessions_dir: std::path::PathBuf,
}

impl OpenCodeConnector {
    pub fn new(sessions_dir: std::path::PathBuf) -> Self {
        Self { sessions_dir }
    }
}

impl HarnessConnector for OpenCodeConnector {
    fn id(&self) -> &str {
        "opencode"
    }

    fn is_present(&self) -> bool {
        self.sessions_dir.exists()
    }

    fn discover(&self, _since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        todo!("OpenCodeConnector::discover")
    }

    fn parse(&self, _r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        todo!("OpenCodeConnector::parse")
    }

    fn read(&self, _locator: &Locator) -> Result<String> {
        todo!("OpenCodeConnector::read")
    }
}
