use crate::models::{Conversation, ConversationRef, Harness, Locator, Message, Role};
use crate::{AgentSessionsError, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

pub fn default_home_dir(subdir: &[&str]) -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    subdir.iter().fold(home, |path, segment| path.join(segment))
}

pub fn discover_jsonl(
    root: &Path,
    harness: &str,
    since: Option<SystemTime>,
) -> Result<Vec<ConversationRef>> {
    if !root.exists() {
        return Ok(vec![]);
    }

    let mut conversations = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        let path = entry.path();
        if !entry.file_type().is_file()
            || path.extension().and_then(|ext| ext.to_str()) != Some("jsonl")
        {
            continue;
        }

        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        let modified_at = metadata.modified().unwrap_or(UNIX_EPOCH);
        if since.is_some_and(|threshold| modified_at <= threshold) {
            continue;
        }

        let id = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("session")
            .to_string();
        let source_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

        conversations.push(ConversationRef {
            id,
            source_path,
            modified_at,
        });
    }

    conversations.sort_by(|a, b| {
        b.modified_at
            .cmp(&a.modified_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    let _ = harness;
    Ok(conversations)
}

pub fn parse_jsonl_session(
    ref_: &ConversationRef,
    harness: Harness,
) -> Result<(Conversation, Vec<Message>)> {
    let raw = fs::read_to_string(&ref_.source_path)?;
    let mut conversation_id = ref_.id.clone();
    let mut harness_version = None;
    let mut project_path = None;
    let mut repo_url = None;
    let mut git_branch = None;
    let mut title = None;
    let mut started_at = None;
    let mut messages = Vec::new();

    for line in raw.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => continue,
        };

        if let Some(id) = extract_string(&value, &["conversation_id", "session_id", "id"]) {
            conversation_id = id;
        }
        harness_version = first_some(
            harness_version,
            extract_string(&value, &["harness_version", "version", "runtime_version"]),
        );
        project_path = first_some(
            project_path,
            extract_string(&value, &["project_path", "project", "cwd"]),
        );
        repo_url = first_some(
            repo_url,
            extract_string(&value, &["repo_url", "repository", "git_url"]),
        );
        git_branch = first_some(
            git_branch,
            extract_string(&value, &["git_branch", "branch"]),
        );
        title = first_some(title, extract_string(&value, &["title", "name", "prompt"]));
        started_at = first_some(
            started_at,
            extract_timestamp(&value, &["started_at", "created_at", "timestamp"]),
        );

        if is_conversation_record(&value) {
            continue;
        }

        if let Some(message) = build_message(
            &value,
            &conversation_id,
            messages.len(),
            ref_.source_path.to_string_lossy().to_string(),
            harness.as_str(),
        ) {
            started_at = started_at.or(message.timestamp.clone());
            messages.push(message);
        }
    }

    let conversation = Conversation {
        id: conversation_id,
        harness,
        harness_version,
        project_path,
        repo_url,
        git_branch,
        title,
        started_at,
        source_path: ref_.source_path.to_string_lossy().to_string(),
        message_count: messages.len(),
    };

    Ok((conversation, messages))
}

pub fn read_message_text(locator: &Locator) -> Result<String> {
    let path = Path::new(&locator.source_path);
    let raw = fs::read_to_string(path)?;
    let mut index = 0usize;

    for line in raw.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let value: Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) => continue,
        };

        if is_conversation_record(&value) {
            continue;
        }

        if index == locator.message_ordinal {
            return extract_text(&value).ok_or_else(|| {
                AgentSessionsError::Parse(format!(
                    "message {} has no text",
                    locator.message_ordinal
                ))
            });
        }
        index += 1;
    }

    Err(AgentSessionsError::NotFound(format!(
        "{}:{}",
        locator.source_path, locator.message_ordinal
    )))
}

fn build_message(
    value: &Value,
    conversation_id: &str,
    message_ordinal: usize,
    source_path: String,
    harness: &str,
) -> Option<Message> {
    let role = extract_role(value);
    let model = extract_string(value, &["model", "model_name"]);
    let timestamp = extract_timestamp(value, &["timestamp", "created_at", "time"]);
    let locator = Locator {
        conversation_id: conversation_id.to_string(),
        message_ordinal,
        chunk_ordinal: 0,
        source_path,
        harness: harness.to_string(),
    };

    Some(Message {
        locator,
        role,
        model,
        timestamp,
    })
}

fn is_conversation_record(value: &Value) -> bool {
    matches!(
        extract_string(value, &["type", "kind"]).as_deref(),
        Some("conversation") | Some("session") | Some("meta")
    )
}

fn extract_role(value: &Value) -> Role {
    match extract_string(value, &["role", "speaker", "author"]).as_deref() {
        Some("assistant") => Role::Assistant,
        Some("system") => Role::System,
        _ => Role::User,
    }
}

fn extract_text(value: &Value) -> Option<String> {
    if let Some(text) = extract_string(value, &["content", "text", "message"]) {
        return Some(text);
    }

    if let Some(items) = value.get("content").and_then(|content| content.as_array()) {
        let mut text = String::new();
        for item in items {
            if let Some(piece) = item.as_str() {
                text.push_str(piece);
            } else if let Some(piece) = item.get("text").and_then(|text| text.as_str()) {
                text.push_str(piece);
            }
        }
        if !text.is_empty() {
            return Some(text);
        }
    }

    if let Some(message) = value.get("message") {
        if let Some(text) = extract_text(message) {
            return Some(text);
        }
    }

    None
}

fn extract_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(key).and_then(value_to_string))
        .or_else(|| {
            value
                .get("message")
                .and_then(|message| extract_string(message, keys))
        })
}

fn value_to_string(value: &Value) -> Option<String> {
    value
        .as_str()
        .map(ToString::to_string)
        .or_else(|| value.as_i64().map(|value| value.to_string()))
        .or_else(|| value.as_u64().map(|value| value.to_string()))
        .or_else(|| value.as_f64().map(|value| value.to_string()))
}

fn extract_timestamp(value: &Value, keys: &[&str]) -> Option<DateTime<Utc>> {
    extract_string(value, keys).and_then(|value| {
        DateTime::parse_from_rfc3339(&value)
            .ok()
            .map(|timestamp| timestamp.with_timezone(&Utc))
    })
}

fn first_some<T>(current: Option<T>, candidate: Option<T>) -> Option<T> {
    current.or(candidate)
}
