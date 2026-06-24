use agent_sessions::{Config, Harness, ListQuery, SessionIndex};
use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "agent-sessions",
    about = "Search your local coding-agent sessions by meaning or regex",
    long_about = "agent-sessions indexes your local coding-agent sessions (Claude, Codex, Cursor, \
                  OpenCode) into a local SQLite database and lets you search them by meaning, \
                  keyword, or vector similarity.\n\n\
                  PRIVACY: only vectors + locators are stored — your transcript text is never \
                  copied into the index. Embeddings run locally; no network call is made during \
                  sync or search.\n\n\
                  AGENT USAGE: every command supports --json for stable machine-readable output. \
                  Pipe to jq, or consume from any language. Schema mirrors the Rust library types.\n\n\
                  EXAMPLES:\n  \
                    agent-sessions sync\n  \
                    agent-sessions search \"how did I handle auth\" --json\n  \
                    agent-sessions grep \"NEVER commit\" --json | jq '.[].snippet'\n  \
                    agent-sessions ls --harness claude --limit 5 --json\n  \
                    agent-sessions show <id> --json",
    version
)]
struct Cli {
    /// Emit JSON instead of human-readable text (stable schema, safe to pipe/parse).
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Discover and index new sessions (incremental — skips already-indexed conversations).
    ///
    /// Reads session files from each detected harness, extracts user messages,
    /// chunks and embeds them locally, and stores vectors + locators in ~/.agent-sessions/index.db.
    /// Transcript text is never written to the index.
    ///
    /// EXAMPLES:
    ///   agent-sessions sync
    ///   agent-sessions sync --json          # machine-readable SyncReport
    Sync,

    /// Hybrid semantic + keyword search — the best default for most queries.
    ///
    /// Embeds the query locally, retrieves the top vector candidates, re-ranks
    /// with keyword overlap using Reciprocal Rank Fusion (RRF, k=60), and returns
    /// results sorted by fused score. Slower than grep (reads source files for
    /// re-ranking) but more relevant than either method alone.
    ///
    /// Output fields: locator (conversation_id, message_ordinal, chunk_ordinal,
    /// source_path, harness), score (0.0–1.0), snippet (verbatim message excerpt).
    ///
    /// EXAMPLES:
    ///   agent-sessions search "how did I handle authentication"
    ///   agent-sessions search "database migration pattern" --harness claude --json
    ///   agent-sessions search "error handling" --json | jq '.[0].snippet'
    Search {
        /// Natural-language query — no special syntax needed.
        query: String,
        /// Maximum results to return.
        #[arg(short, long, default_value = "10")]
        limit: usize,
        /// Restrict to one harness: claude, codex, cursor, opencode.
        #[arg(long)]
        harness: Option<String>,
    },

    /// Regex search across all indexed sessions (reads source files — exact matches only).
    ///
    /// For each indexed conversation, reads every user message through the
    /// connector and checks it against the pattern. Returns lines that match.
    /// Use for exact phrases, identifiers, or patterns you know appear verbatim.
    ///
    /// EXAMPLES:
    ///   agent-sessions grep "use pnpm"
    ///   agent-sessions grep "NEVER commit" --json | jq '.[].snippet'
    ///   agent-sessions grep "fn handle_" --harness claude
    Grep {
        /// Regular expression (Rust regex syntax).
        pattern: String,
        /// Restrict to one harness: claude, codex, cursor, opencode.
        #[arg(long)]
        harness: Option<String>,
    },

    /// Pure vector nearest-neighbor search — semantic only, no keyword re-ranking.
    ///
    /// Faster than `search` (no source file reads) but misses exact keyword matches
    /// that aren't semantically close. Use when you want "things like this concept"
    /// rather than "things containing these words".
    ///
    /// EXAMPLES:
    ///   agent-sessions similar "debugging async rust"
    ///   agent-sessions similar "CSS layout tricks" --json
    Similar {
        /// Natural-language query.
        query: String,
        /// Maximum results to return.
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// List indexed conversations with optional filters.
    ///
    /// Returns conversation metadata: id, harness, project_path, started_at,
    /// message_count. Use `show <id>` to inspect a specific conversation.
    ///
    /// EXAMPLES:
    ///   agent-sessions ls
    ///   agent-sessions ls --harness claude --limit 5 --json
    ///   agent-sessions ls --project /Users/me/workspace/myapp --json
    Ls {
        /// Filter by harness: claude, codex, cursor, opencode.
        #[arg(long)]
        harness: Option<String>,
        /// Filter by project path (exact match on stored cwd).
        #[arg(long)]
        project: Option<String>,
        /// Maximum results to return.
        #[arg(short, long, default_value = "20")]
        limit: usize,
        /// Offset for pagination.
        #[arg(long, default_value = "0")]
        offset: usize,
    },

    /// Show full metadata for a specific conversation by ID.
    ///
    /// The ID comes from `ls --json` or a search hit's `locator.conversation_id`.
    ///
    /// EXAMPLES:
    ///   agent-sessions show abc123
    ///   agent-sessions show abc123 --json
    Show {
        /// Conversation ID (from `ls` or a search hit locator).
        id: String,
    },

    /// List all registered harness connectors and whether their data is present.
    ///
    /// Output fields: id (harness name), present (bool — data directory exists).
    /// Use this to diagnose why a harness isn't being synced.
    ///
    /// EXAMPLES:
    ///   agent-sessions harnesses
    ///   agent-sessions harnesses --json
    Harnesses,

    /// Pre-fetch the local embedding model weights (one-time, ~30 MB).
    ///
    /// On first sync/search the model is downloaded automatically. Run `setup`
    /// explicitly for CI environments, Docker builds, or air-gapped machines.
    ///
    /// EXAMPLES:
    ///   agent-sessions setup
    Setup,

    /// Manage scheduled background sync (launchd / systemd --user / Scheduled Task).
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
    },

    /// Install SKILL.md into agent skill directories for agent discoverability.
    ///
    /// Symlinks the bundled SKILL.md into ~/.claude/skills/, .cursor/skills/,
    /// etc. so coding agents can auto-discover how to use this tool.
    ///
    /// EXAMPLES:
    ///   agent-sessions skill install
    Skill {
        /// Skill action (currently: install).
        name: Option<String>,
    },
}

#[derive(Subcommand)]
enum ScheduleAction {
    /// Add a new scheduled sync.
    Add {
        /// Cron expression (e.g. "0 * * * *").
        #[arg(long)]
        cron: String,
    },
    /// Remove the scheduled sync.
    Remove,
    /// Show the current schedule.
    Status,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut index = SessionIndex::open(Config::default())?;

    match cli.command {
        Commands::Sync => {
            let report = index.sync()?;
            print_json_or_text(cli.json, &report, |report| {
                format!(
                    "indexed {} conversations, {} chunks",
                    report.conversations_indexed, report.chunks_added
                )
            })?;
        }
        Commands::Search { query, limit, harness: _ } => {
            let mut hits = index.search(&query)?;
            hits.truncate(limit);
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                for h in &hits {
                    println!("[{:.3}] {} (msg {})", h.score, h.locator.conversation_id, h.locator.message_ordinal);
                    println!("  {}", h.snippet.lines().next().unwrap_or("").trim());
                }
                if hits.is_empty() { println!("no results"); }
            }
        }
        Commands::Grep { pattern, harness: _ } => {
            let hits = index.grep(&pattern)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                for h in &hits {
                    println!("{} (msg {}): {}", h.locator.conversation_id, h.locator.message_ordinal, h.snippet.trim());
                }
                if hits.is_empty() { println!("no results"); }
            }
        }
        Commands::Similar { query, limit } => {
            let mut hits = index.similar(&query)?;
            hits.truncate(limit);
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&hits)?);
            } else {
                for h in &hits {
                    println!("[{:.3}] {} (msg {})", h.score, h.locator.conversation_id, h.locator.message_ordinal);
                    println!("  {}", h.snippet.lines().next().unwrap_or("").trim());
                }
                if hits.is_empty() { println!("no results"); }
            }
        }
        Commands::Ls { harness, project, limit, offset } => {
            let query = ListQuery {
                harness: harness.as_deref().map(Harness::from_str),
                project_path: project,
                limit: Some(limit),
                offset: Some(offset),
            };
            let page = index.list_conversations(query)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&page.items)?);
            } else {
                for c in &page.items {
                    println!(
                        "{} [{}] msgs={} {}",
                        c.id, c.harness.as_str(), c.message_count,
                        c.project_path.as_deref().unwrap_or("")
                    );
                }
                if page.items.is_empty() { println!("no conversations indexed — run sync first"); }
            }
        }
        Commands::Show { id } => {
            let conversation = index.get_conversation(&id)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&conversation)?);
            } else {
                match conversation {
                    Some(c) => println!("{:#?}", c),
                    None => { eprintln!("not found: {id}"); std::process::exit(1); }
                }
            }
        }
        Commands::Harnesses => {
            let harnesses = index.harnesses();
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&harnesses)?);
            } else {
                for h in &harnesses {
                    println!("{:<12} {}", h.id, if h.present { "present" } else { "absent" });
                }
            }
        }
        Commands::Setup => not_yet("setup"),
        Commands::Schedule { action } => match action {
            ScheduleAction::Add { cron } => not_yet(&format!("schedule add {cron}")),
            ScheduleAction::Remove => not_yet("schedule remove"),
            ScheduleAction::Status => not_yet("schedule status"),
        },
        Commands::Skill { name } => not_yet(name.as_deref().unwrap_or("skill")),
    }

    Ok(())
}

fn print_json_or_text<T, F>(json: bool, value: &T, fallback: F) -> Result<()>
where
    T: serde::Serialize,
    F: FnOnce(&T) -> String,
{
    if json {
        println!("{}", serde_json::to_string_pretty(value)?);
    } else {
        println!("{}", fallback(value));
    }
    Ok(())
}

fn not_yet(command: &str) -> ! {
    eprintln!("{command} not yet implemented");
    std::process::exit(1);
}
