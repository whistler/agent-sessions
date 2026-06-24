use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Locator, Message};
use crate::Result;

/// Reads OpenAI Codex / codex-cli session transcripts.
pub struct CodexConnector {
    sessions_dir: std::path::PathBuf,
}

impl CodexConnector {
    pub fn new(sessions_dir: std::path::PathBuf) -> Self {
        Self { sessions_dir }
    }
}

impl HarnessConnector for CodexConnector {
    fn id(&self) -> &str {
        "codex"
    }

    fn is_present(&self) -> bool {
        self.sessions_dir.exists()
    }

    fn discover(&self, _since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        todo!("CodexConnector::discover")
    }

    fn parse(&self, _r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        todo!("CodexConnector::parse")
    }

    fn read(&self, _locator: &Locator) -> Result<String> {
        todo!("CodexConnector::read")
    }
}
