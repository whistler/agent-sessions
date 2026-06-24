use crate::connector::HarnessConnector;
use crate::models::{Conversation, ConversationRef, Locator, Message};
use crate::Result;

/// Reads Claude Code / Claude Agent SDK session transcripts.
pub struct ClaudeConnector {
    sessions_dir: std::path::PathBuf,
}

impl ClaudeConnector {
    pub fn new(sessions_dir: std::path::PathBuf) -> Self {
        Self { sessions_dir }
    }

    /// Returns the default sessions directory (~/.claude/sessions).
    pub fn default_sessions_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_default();
        std::path::PathBuf::from(home).join(".claude").join("sessions")
    }
}

impl HarnessConnector for ClaudeConnector {
    fn id(&self) -> &str {
        "claude"
    }

    fn is_present(&self) -> bool {
        self.sessions_dir.exists()
    }

    fn discover(&self, _since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>> {
        todo!("ClaudeConnector::discover")
    }

    fn parse(&self, _r: &ConversationRef) -> Result<(Conversation, Vec<Message>)> {
        todo!("ClaudeConnector::parse")
    }

    fn read(&self, _locator: &Locator) -> Result<String> {
        todo!("ClaudeConnector::read")
    }
}
