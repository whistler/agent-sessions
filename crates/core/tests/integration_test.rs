use agent_sessions::connectors::{
    ClaudeConnector, CodexConnector, CursorConnector, OpenCodeConnector,
};
use agent_sessions::store::sqlite_vec::SqliteVecStore;
use agent_sessions::{DefaultChunker, HarnessConnector, HashEmbedder, ListQuery, SessionIndex};
use std::path::PathBuf;

fn fixture_root(parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    for part in parts {
        path.push(part);
    }
    path
}

fn test_index() -> SessionIndex {
    let store = Box::new(SqliteVecStore::open_in_memory().unwrap());
    let embedder = Box::new(HashEmbedder::new(64));
    let chunker = Box::new(DefaultChunker::default());
    let connectors = vec![
        Box::new(ClaudeConnector::new(fixture_root(&["claude", "projects"])))
            as Box<dyn agent_sessions::HarnessConnector>,
        Box::new(CodexConnector::new(fixture_root(&["codex", "sessions"])))
            as Box<dyn agent_sessions::HarnessConnector>,
        Box::new(CursorConnector::new(fixture_root(&["cursor", "sessions"])))
            as Box<dyn agent_sessions::HarnessConnector>,
        Box::new(OpenCodeConnector::new(fixture_root(&[
            "opencode", "sessions",
        ]))) as Box<dyn agent_sessions::HarnessConnector>,
    ];

    SessionIndex::from_parts(store, embedder, chunker, connectors)
}

#[test]
fn connector_discovers_and_reads_fixture_text() {
    let connector = ClaudeConnector::new(fixture_root(&["claude", "projects"]));
    let refs = connector.discover(None).unwrap();
    assert_eq!(refs.len(), 1);

    let (conversation, messages) = connector.parse(&refs[0]).unwrap();
    assert_eq!(conversation.id, "abc123");
    assert_eq!(conversation.message_count, 2);

    let text = connector.read(&messages[0].locator).unwrap();
    assert!(text.contains("read before editing"));
}

#[test]
fn sync_search_grep_and_list_work() {
    let mut index = test_index();
    let report = index.sync().unwrap();
    assert_eq!(report.conversations_indexed, 2);
    assert_eq!(report.chunks_added, 4);
    assert!(report.harness_errors.is_empty());

    let search_hits = index.search("read before editing").unwrap();
    assert!(!search_hits.is_empty());
    assert!(
        search_hits[0]
            .snippet
            .to_lowercase()
            .contains("read before editing")
    );

    let similar_hits = index.similar("read before editing").unwrap();
    assert!(!similar_hits.is_empty());

    let grep_hits = index.grep("pnpm").unwrap();
    assert!(!grep_hits.is_empty());

    let page = index
        .list_conversations(ListQuery {
            harness: None,
            project_path: Some("/workspace/test-project".to_string()),
            limit: Some(10),
            offset: Some(0),
        })
        .unwrap();
    assert_eq!(page.total, 2);
    assert_eq!(page.items.len(), 2);
}

#[test]
fn read_returns_full_message_text() {
    let mut index = test_index();
    index.sync().unwrap();
    let conversation = index.get_conversation("abc123").unwrap().unwrap();
    let locator = agent_sessions::Locator {
        conversation_id: conversation.id,
        message_ordinal: 0,
        chunk_ordinal: 0,
        source_path: conversation.source_path,
        harness: conversation.harness.as_str().to_string(),
    };
    let text = index.read(&locator).unwrap();
    assert!(text.contains("Please read before editing"));
}
