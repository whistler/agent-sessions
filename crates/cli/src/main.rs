use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "agent-sessions",
    about = "Search your local coding-agent sessions by meaning or regex",
    version
)]
struct Cli {
    /// Output JSON instead of human-readable text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Sync new sessions into the local index.
    Sync,
    /// Semantic search across all indexed sessions.
    Search {
        /// The query string.
        query: String,
        /// Maximum number of results.
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Regex search across all indexed sessions.
    Grep {
        /// The regex pattern.
        pattern: String,
    },
    /// Find sessions similar to a given query.
    Similar {
        /// The query string.
        query: String,
        /// Maximum number of results.
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// List indexed conversations.
    Ls {
        /// Filter by harness (claude, codex, cursor, opencode).
        #[arg(long)]
        harness: Option<String>,
        /// Filter by project path.
        #[arg(long)]
        project: Option<String>,
        /// Maximum number of results.
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },
    /// Show a specific conversation.
    Show {
        /// Conversation ID.
        id: String,
    },
    /// List registered harness connectors and their status.
    Harnesses,
    /// Interactive setup wizard.
    Setup,
    /// Manage scheduled sync jobs.
    Schedule {
        #[command(subcommand)]
        action: ScheduleAction,
    },
    /// Run or manage skills.
    Skill {
        /// Skill name to run.
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

fn not_yet() -> ! {
    eprintln!("not yet implemented");
    std::process::exit(1);
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Sync => not_yet(),
        Commands::Search { .. } => not_yet(),
        Commands::Grep { .. } => not_yet(),
        Commands::Similar { .. } => not_yet(),
        Commands::Ls { .. } => not_yet(),
        Commands::Show { .. } => not_yet(),
        Commands::Harnesses => not_yet(),
        Commands::Setup => not_yet(),
        Commands::Schedule { .. } => not_yet(),
        Commands::Skill { .. } => not_yet(),
    }
}
