use crate::models::{Conversation, ConversationRef, Locator, Message};
use crate::Result;

pub trait HarnessConnector: Send + Sync {
    fn id(&self) -> &str;
    fn is_present(&self) -> bool;
    fn discover(&self, since: Option<std::time::SystemTime>) -> Result<Vec<ConversationRef>>;
    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)>;
    fn read(&self, locator: &Locator) -> Result<String>;
}
