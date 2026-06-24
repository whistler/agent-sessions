/// Real-data smoke tests — require actual harness data on disk.
/// Run with: cargo test -p agent-sessions real_data -- --include-ignored
///
/// Each test is self-skipping: if the harness isn't present it passes silently.
/// They are `#[ignore]` so `cargo test` (CI) doesn't fail on a clean machine.
use agent_sessions::{
    connector::HarnessConnector,
    connectors::{ClaudeConnector, CodexConnector, CursorConnector, OpenCodeConnector},
};

fn home() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

// ─── Claude ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires ~/.claude/projects"]
fn claude_discovers_real_sessions() {
    let c = ClaudeConnector::new(home().join(".claude").join("projects"));
    if !c.is_present() {
        eprintln!("skip: ~/.claude/projects not found");
        return;
    }
    let refs = c.discover(None).unwrap();
    assert!(!refs.is_empty(), "expected at least one Claude session");
    eprintln!("claude: {} sessions discovered", refs.len());
}

#[test]
#[ignore = "requires ~/.claude/projects"]
fn claude_parses_first_session() {
    let c = ClaudeConnector::new(home().join(".claude").join("projects"));
    if !c.is_present() { return; }
    let refs = c.discover(None).unwrap();
    if refs.is_empty() { return; }

    let (conv, messages) = c.parse(&refs[0]).unwrap();
    assert!(!conv.id.is_empty(), "conversation id must be non-empty");
    assert!(!messages.is_empty(), "must have at least one user message");
    eprintln!(
        "claude: conv {} — {} messages, project {:?}",
        conv.id, conv.message_count, conv.project_path
    );
}

#[test]
#[ignore = "requires ~/.claude/projects"]
fn claude_reads_first_message() {
    let c = ClaudeConnector::new(home().join(".claude").join("projects"));
    if !c.is_present() { return; }
    let refs = c.discover(None).unwrap();
    if refs.is_empty() { return; }
    let (_, messages) = c.parse(&refs[0]).unwrap();
    if messages.is_empty() { return; }

    let text = c.read(&messages[0].locator).unwrap();
    assert!(!text.trim().is_empty(), "message text must be non-empty");
    eprintln!("claude: first message ({} chars): {}…", text.len(), &text[..text.len().min(80)]);
}

// ─── Codex ───────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires ~/.codex/sessions"]
fn codex_discovers_real_sessions() {
    let c = CodexConnector::new(home().join(".codex").join("sessions"));
    if !c.is_present() {
        eprintln!("skip: ~/.codex/sessions not found");
        return;
    }
    let refs = c.discover(None).unwrap();
    assert!(!refs.is_empty(), "expected at least one Codex session");
    eprintln!("codex: {} sessions discovered", refs.len());
}

#[test]
#[ignore = "requires ~/.codex/sessions"]
fn codex_parses_first_session() {
    let c = CodexConnector::new(home().join(".codex").join("sessions"));
    if !c.is_present() { return; }
    let refs = c.discover(None).unwrap();
    if refs.is_empty() { return; }

    let (conv, messages) = c.parse(&refs[0]).unwrap();
    assert!(!conv.id.is_empty());
    eprintln!("codex: conv {} — {} messages", conv.id, messages.len());
}

// ─── Cursor ──────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires ~/.cursor agent-transcripts"]
fn cursor_discovers_real_sessions() {
    let c = CursorConnector::new(CursorConnector::default_sessions_dir());
    if !c.is_present() {
        eprintln!("skip: cursor sessions not found");
        return;
    }
    let refs = c.discover(None).unwrap();
    assert!(!refs.is_empty());
    eprintln!("cursor: {} sessions discovered", refs.len());
}

#[test]
#[ignore = "requires ~/.cursor agent-transcripts"]
fn cursor_parses_first_session() {
    let c = CursorConnector::new(CursorConnector::default_sessions_dir());
    if !c.is_present() { return; }
    let refs = c.discover(None).unwrap();
    if refs.is_empty() { return; }

    let (conv, messages) = c.parse(&refs[0]).unwrap();
    assert!(!conv.id.is_empty());
    eprintln!("cursor: conv {} — {} messages, title {:?}", conv.id, messages.len(), conv.title);
}

// ─── OpenCode ────────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires ~/.local/share/opencode/opencode.db"]
fn opencode_discovers_real_sessions() {
    let c = OpenCodeConnector::new(OpenCodeConnector::default_sessions_dir());
    if !c.is_present() {
        eprintln!("skip: opencode db not found");
        return;
    }
    let refs = c.discover(None).unwrap();
    assert!(!refs.is_empty());
    eprintln!("opencode: {} sessions discovered", refs.len());
}

// ─── End-to-end ──────────────────────────────────────────────────────────────

#[test]
#[ignore = "requires ~/.claude/projects — runs full sync+search pipeline"]
fn end_to_end_sync_and_search() {
    use agent_sessions::{Config, SessionIndex};

    let tmp = tempfile::tempdir().unwrap();
    let mut idx = SessionIndex::open(Config {
        store_path: Some(tmp.path().join("index.db")),
        ..Default::default()
    })
    .unwrap();

    let report = idx.sync().unwrap();
    eprintln!(
        "synced {} conversations, {} chunks, errors: {:?}",
        report.conversations_indexed, report.chunks_added, report.harness_errors
    );

    if report.conversations_indexed == 0 {
        eprintln!("no sessions found — skipping search assertions");
        return;
    }

    // search should return at least one result
    let hits = idx.search("how do I").unwrap();
    assert!(!hits.is_empty(), "expected search results");
    eprintln!("search 'how do I': {} hits, top score {:.4}", hits.len(), hits[0].score);

    // grep should work
    let grep_hits = idx.grep("fn ").unwrap();
    eprintln!("grep 'fn ': {} hits", grep_hits.len());
}
