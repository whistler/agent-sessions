# agent-sessions — Library Design

*Working name: `agent-sessions` (import `agent_sessions`, CLI `agent-sessions`). Lock the name before first publish — see [Naming](#naming).*

**What it is:** a standalone, dependency-light, **local-first** library + CLI that discovers, normalizes, embeds, and searches your coding-agent sessions across harnesses (Claude, Codex, Cursor, OpenCode). It stands entirely on its own as "search my coding history" — no external service, no intelligence layer baked in.

**Prime directive — portability.** A single developer should be able to install it with one command, run it on macOS / Linux / Windows with zero config, and end up with **one portable index file** they can copy, back up, inspect, or delete. No server, no daemon required, no cloud.

**Privacy invariant (non-negotiable):** the index stores **vectors + locators only — never transcript text**. Source session files remain the single source of truth; message text is read lazily on demand and never persisted. This is testable (see [AT-PRIV](#acceptance-tests)) and currently *violated* by the prototype store — fixing it is prerequisite to publishing.

---

## Public API

The library is the contract; the CLI is a thin shell that calls the identical methods. Anything the CLI can do, the library can do.

```rust
let mut idx = SessionIndex::open(Default::default())?;   // zero config: auto-detect harnesses, local embedder, default db
let report = idx.sync()?;                                 // discover -> parse -> chunk -> embed -> store (incremental, idempotent)

idx.search("read before editing")?;                       // hybrid — the do-what-I-mean default
idx.grep(r"use pnpm")?;                                    // keyword / regex — exact strings
idx.similar("read before editing")?;                      // pure vector — nearest neighbors

idx.list_conversations(ListQuery { harness, since, until, limit, cursor, .. })?;  // time filter + keyset paging
idx.get_conversation(&id)?;                               // full, lazily read
idx.read(&locator)?;                                      // lazy single-message text
idx.harnesses()?;                                         // enabled · present · counts
idx.register(Box::new(MyConnector));                      // third-party harness, no fork
```

### Search: three verbs, one default

Not modes on one function, and not one combined function — **three honest verbs**, each a sink, with `search` as the default people reach for.

| Verb | Mechanism | When | Cost |
|---|---|---|---|
| `search` (default) | vector recall, then **RRF**-fused with keyword rank over the recalled candidates | "find sessions about X" — the smart default | ~vector + cheap re-rank of K candidates |
| `grep` | regex/literal, lazily reads candidate text via `connector.read()` | exact strings, identifiers | I/O-bound (reads text); slowest at scale |
| `similar` | vector KNN over chunk embeddings | pure nearest-neighbor, no keyword | one query embed + KNN (C-fast to ~1M) |

**Why `search` (hybrid) is the default — not because it's fastest.** You rarely remember the exact words from a past session; you remember the gist ("that time I told it to read first"). Semantic recall finds it from a paraphrase; keyword can't. Hybrid adds exact-match precision on top, and is only marginally costlier than pure vector because the keyword step re-ranks the **small recalled candidate set**, not the whole corpus. Pure `grep` is actually the *slowest* path here, because nothing is stored as text — a global keyword scan reads files through connectors.

RRF (Reciprocal Rank Fusion) is chosen over weighted score blending: no tunable weights, robust to the two scorers being on different scales — `score(d) = Σ 1/(k + rank_i(d))`, default `k = 60`.

Doc caveat to surface: ripgrep-brained users may expect bare `search` to be keyword. Document that `search` is fuzzy/semantic by default and `grep` is the literal path.

### Pagination & time filtering

Keyset (cursor) pagination, not offset — stable while sync writes concurrently. `cursor` is an opaque token of `(started_at, id)`. `since`/`until` filter `Conversation.started_at`; the CLI accepts human durations (`7d`, `24h`) and ISO timestamps.

---

## Extension points

```rust
trait HarnessConnector {
    fn id(&self) -> &str;
    fn is_present(&self) -> bool;
    fn discover(&self, since: Option<SystemTime>) -> Result<Vec<ConversationRef>>;
    fn parse(&self, r: &ConversationRef) -> Result<(Conversation, Vec<Message>)>;
    fn read(&self, locator: &Locator) -> Result<String>;   // lazy text — the only place text is materialized
}

trait Embedder { fn id(&self) -> &str; fn dim(&self) -> usize; fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>; }
trait Chunker  { fn chunk(&self, text: &str) -> Vec<String>; }            // deterministic; short text -> [text]
trait VectorStore {                                                        // the portability seam (see Storage)
    fn upsert_conversation(&mut self, c: &Conversation) -> Result<()>;
    fn upsert_chunks(&mut self, conv_id: &str, chunks: &[ChunkVector]) -> Result<()>;  // NO text field
    fn vector_search(&self, vec: &[f32], limit: usize, where_: &Filter) -> Result<Vec<StoredChunk>>;
    fn meta(&self) -> Result<Meta>;
}
```

`ChunkVector` carries `locator`, `vector`, `role`, and filterable scalars (`harness`, `project_path`, `model`, `timestamp`) — **not** `text`. Text lives only in memory during indexing and is re-read on demand.

---

## CLI

```
agent-sessions sync   [--since 7d] [--watch]
agent-sessions search "<q>" [--harness H] [--project P] [--since 7d] [--limit N] [--json]
agent-sessions grep   "<regex>" [--harness H] [--limit N] [--json]
agent-sessions similar "<q>" [--harness H] [--limit N] [--json]
agent-sessions ls     [--harness H] [--project P] [--since 7d] [--until now] [--limit N] [--cursor TOK] [--json]
agent-sessions show   <conversation_id> [--json]
agent-sessions harnesses [--json]
agent-sessions setup                                  # eagerly fetch local runtime + default model (see Distribution)
agent-sessions models pull <id>
agent-sessions schedule enable|disable [--interval 30m]
agent-sessions skill install                          # symlink SKILL.md into agent skill dirs
```

`--json` on every read command emits the same DTOs the library returns (one schema, defined once). The CLI owns config-file discovery and pretty-printing; the library takes explicit args and returns data.

### CLI documentation for agents

Coding agents are a primary consumer. The CLI is designed to be agent-readable:

- Every command has a `--json` flag for machine-readable output (stable schema).
- `agent-sessions --help` and `agent-sessions <cmd> --help` use clap's structured help — agents can read it directly.
- `SKILL.md` at repo root: agent-facing usage doc (discovery patterns, search idioms, privacy guarantee). `agent-sessions skill install` symlinks it into `~/.claude/skills/`, `.cursor/skills/`, etc.
- `llms.txt` + `llms-full.txt` at the repo root for LLM discoverability (spec at llmstxt.org).

Example agent usage pattern:
```bash
# Sync then search — works in any coding agent shell
agent-sessions sync
agent-sessions search "how did I handle auth" --json | jq '.[0].snippet'
agent-sessions grep "NEVER commit" --json
```

---

## Configuration

Layering: **explicit args > config file > defaults.** The library never reads the file itself — the CLI loads it and passes explicit args, keeping the library a pure sink. File at `~/.config/agent-sessions/config.toml` (XDG; `%APPDATA%` on Windows), overridable with `--config`.

```toml
[index]
db_backend = "sqlite-vec"          # "sqlite-vec" (default, portable) | "lancedb" (scale)
db_path    = "~/.agent-sessions/index.db"
embedder   = "local/BAAI/bge-small-en-v1.5"   # or "api/voyage/...", "api/openai/..."
roles      = ["user"]              # future: ["user", "assistant"]

[harnesses]
enabled = ["claude", "codex", "cursor", "opencode"]   # allow-list; omit -> auto-detect all present

[harnesses.cursor]
enabled = false                    # disable one harness

[harnesses.codex]
path = "~/.codex"                  # override autodetected base path

[sync]
auto     = false
interval = "30m"
```

Disabling a harness = remove from `enabled`, or set `enabled = false` under its section. Auto-detect remains the zero-config default when `[harnesses].enabled` is omitted.

---

## Storage: sqlite-vec vs LanceDB

The store is behind the `VectorStore` trait so the backend is a swap, not a rewrite. **Default = sqlite-vec** (max portability), **opt-in = LanceDB** (scale). Both satisfy the no-text invariant. Both are Rust-native or Rust-loadable, which reinforces the language choice below.

| Dimension | **sqlite-vec** (default) | **LanceDB** (opt-in) |
|---|---|---|
| On-disk shape | **single `.db` file** — copy/back up/delete trivially | a directory of Lance/Arrow files |
| Embeddability | C extension loaded via `rusqlite`; nothing to run | embedded (Rust-native), no server |
| Packaging weight | tiny (extension only) | heavier (Arrow + Lance) |
| ANN index | exact brute-force KNN — fast in C to ~1M rows | real ANN (IVF_PQ etc.), scales to many millions |
| Metadata filtering | native SQL `WHERE` / joins | predicate pushdown on columns |
| Recall at MVP scale (1M chunks) | exact (100%), linear but C-fast | approximate, sub-linear |
| Backup / portability | `cp index.db` | copy the directory |
| Versioning / time-travel | no | yes (Lance versioning) |
| Maturity | newer, simple surface | newer format, richer surface |

**Decision:** default to **sqlite-vec**. At the [scalability target](#scalability) — one user, ≤1M chunks, 384-dim — exact brute-force in C is fast enough, recall is perfect, and "one file you can copy" is the strongest portability story. LanceDB is the documented escape hatch beyond a few million chunks or when ANN/time-travel is needed.

---

## Language: Rust core, multi-language faces

**Decision: a Rust core**, exposed through two faces from one codebase. This is driven by (1) the desktop app is **Tauri** (Rust shell) — a Rust library compiles straight in with no sidecar; (2) the heavy deps (`lance`, sqlite-vec via `rusqlite`, `fastembed`/`candle`) are Rust-native; (3) the library's logic is simple I/O + data transformation, which bounds Rust's cost; (4) it yields a single static CLI binary — the portability goal.

**One core, two faces:**
- **Tauri desktop app** → `#[tauri::command]` wrappers, no IPC, no second runtime. JS frontend calls `await invoke('search', { q })`.
- **Standalone CLI** → native `clap` binary (single static file).

```rust
// Tauri command — JS frontend: await invoke('search', { q })
#[tauri::command]
async fn search(state: State<'_, Index>, q: String) -> Result<Vec<Hit>, String> {
    state.search(&q).map_err(|e| e.to_string())
}
```

### Three-language tradeoff (for this library, given Tauri + JS frontend)

| | **Rust (chosen)** | Python | JS/TS |
|---|---|---|---|
| Tauri integration | ✅ compiles in as commands, one binary | ❌ sidecar: PyInstaller bundle + localhost HTTP + signing pain | ⚠️ webview/Node, compute blocks UI thread |
| Distribution | single static binary (CLI) + signed app | bundled interpreter (`uv tool` ok for CLI) | npm/npx, but desktop still needs the Rust shell |
| Heavy deps | native: lance, rusqlite+sqlite-vec, fastembed/candle | mature wrappers around the same native libs | lancedb-node; transformers.js slow |
| Local embedding perf | best (direct ONNX/Candle) | good (C extensions do the work) | weakest |
| Dev speed | slowest (lifetimes, ONNX bundling) | fastest | fast |
| Contributor pool | smaller | largest | large |

**Why not Python:** the Tauri **sidecar tax** — PyInstaller bundle, localhost HTTP boundary, macOS notarization friction, ~tens of MB interpreter. Works fine for a pure CLI; prohibitive once a desktop binary is in scope. **Why not JS/TS:** weakest for compute-heavy core; keep it to frontend glue.

**On Rust lifetimes (a real learning-curve cost):** lifetimes annotate how long a reference is valid so the compiler prevents use-after-free without a GC. They bite hardest in zero-copy, heavily-borrowed code. This library mostly sidesteps them by having types **own** their data (`String`, `Vec<f32>`) and cloning at boundaries — at this scale the clones are free relative to disk/embedding I/O. It's an unusually good first Rust project: simple domain logic, hard parts inside mature crates.

### Key Rust crates

| Concern | Crate | Notes |
|---|---|---|
| Chunking | [`text-splitter`](https://crates.io/crates/text-splitter) | char, token (tiktoken-rs / HF tokenizers), markdown, code-aware; zero hand-rolling |
| Embedding (local) | [`fastembed`](https://crates.io/crates/fastembed) | Qdrant-maintained, wraps `ort`; BGE-small/base/large, multilingual-e5; lazy model download |
| ONNX runtime | [`ort`](https://crates.io/crates/ort) | fastembed brings this; `load-dynamic` feature → shared `.dylib/.so/.dll` |
| Vector store | [`rusqlite`](https://crates.io/crates/rusqlite) + [`sqlite-vec`](https://crates.io/crates/sqlite-vec) | bundled SQLite + C extension for exact KNN |
| Embedder (pure-Rust escape hatch) | [`candle`](https://crates.io/crates/candle-core) | HuggingFace; removes native ONNX dep, single static binary |
| CLI | [`clap`](https://crates.io/crates/clap) derive | auto `--help`, structured help, shell completions |
| JSON | [`serde_json`](https://crates.io/crates/serde_json) | `--json` flag output |
| Regex | [`regex`](https://crates.io/crates/regex) | `grep` verb |
| Directory walk | [`walkdir`](https://crates.io/crates/walkdir) | harness discovery |
| Error handling | [`thiserror`](https://crates.io/crates/thiserror) | typed errors; `anyhow` in CLI |

---

## Distribution & runtime

### Per-platform builds

Native binaries are per-platform **and** per-arch — unavoidable for any compiled embedding runtime, in any language. Build the matrix in CI:

| Target | For |
|---|---|
| `aarch64-apple-darwin` | Apple Silicon |
| `x86_64-apple-darwin` | Intel Mac |
| `x86_64-unknown-linux-gnu` | most Linux |
| `aarch64-unknown-linux-gnu` | ARM Linux / cloud |
| `x86_64-pc-windows-msvc` | Windows |

Use **`dist` (cargo-dist)** to generate the binaries, installer scripts, Homebrew formula, npm shim, and `binstall` metadata from one config + a GitHub release.

### Install commands (best UX first)

```bash
curl -fsSL https://github.com/whistler/agent-sessions/releases/latest/download/install.sh | sh
brew install whistler/tap/agent-sessions                # mac/linux (cargo-dist Homebrew formula)
winget install agent-sessions                           # Windows (or: scoop install …)
cargo binstall agent-sessions                           # cargo users, prebuilt
cargo install agent-sessions                            # fallback: from source (needs toolchain)
```

The curl/powershell installer is the headline — generated by `dist` (cargo-dist), hosted on GitHub Releases alongside the binaries, detects platform+arch, pulls the right prebuilt, puts it on PATH, no toolchain required. `cargo install` is the from-source fallback, not the primary path.

### Install is light; provisioning is lazy + explicit

Heavy downloads (the ONNX runtime ~10–20 MB, the embedding model ~30 MB int8 / ~130 MB fp32) are **not** tied to install — there's no reliable cross-platform post-install hook (raw binaries have no install step; `cargo install` has none; only npm has `postinstall`), and many users with an API embedder never need them.

- **First actual use** lazily fetches runtime + weights once, with a progress bar, into a shared cache (`~/.cache/agent-sessions/...`). Subsequent runs are instant and offline.
- **Explicit command** for determinism (CI, Docker layers, air-gapped prep): `agent-sessions setup` / `agent-sessions models pull <id>`. (Playwright's `npx playwright install` precedent.)

### ONNX runtime: shared, reused, optional

- The runtime is a dependency of the **default local embedder only** — the `Embedder` trait is pluggable, so an API embedder pulls no runtime at all.
- Build `ort` with **`load-dynamic`**: the binary links the onnxruntime shared lib at runtime from the shared cache, so the CLI **and** the Tauri app **reuse one copy** instead of vendoring it twice. Costs: first-run fetch, lib/`ort` version matching, macOS dylib signing.
- **Escape hatch:** a pure-Rust embedder via **Candle** removes the native onnxruntime dependency entirely (one static binary, weights still cached). Younger than ONNX Runtime; swapping ORT→Candle is internal to the `Embedder` impl.

Model weights are always a download-once shared cache — never bundled, never duplicated.

---

## Background sync

Sync is incremental and idempotent, so the portable mechanism is **a scheduled invocation of `agent-sessions sync`, not a bespoke daemon.** Three tiers:

1. **Manual** — `agent-sessions sync`. Default.
2. **Foreground watcher** — `sync --watch`: long-lived process re-syncing on an interval (later: filesystem events). Any host process can run this loop in-process via the library API.
3. **OS scheduler (set-and-forget, fully portable)** — `schedule enable [--interval 30m]` installs the native unit: macOS `launchd` LaunchAgent · Linux `systemd --user` timer (fallback cron) · Windows Scheduled Task. `schedule disable` removes it.

Enable via `[sync] auto = true` + `schedule enable`, or just `--watch`. Optional lazy **sync-on-read** (config-gated, off by default): if the last sync is older than `interval`, `search`/`ls` trigger a quick incremental sync first.

---

## Naming

- **Package / import:** `agent-sessions` / `agent_sessions`. Alternatives considered: `agent-logs`/`agentlog` (free), `ai-sessions` (vague), `session-history` (collides with web sessions). Rejected `agent-traces`/`agent-runs` (observability/CI connotations), `harness-sessions` (insider jargon).
- **Namespacing (`@scope/name`)** only exists in **npm** (scoped packages, free to claim). **PyPI** and **crates.io** are flat — no `@vendor` prefix; PEP 420 namespace packages affect import paths only, not ownership. Don't tie the identity to a personal handle (`@whistler/…`); a neutral brand org ages better. If only PyPI/crates, just claim the clean flat name.
- **Renaming later is feasible but a breaking change** — registries don't do in-place renames. Pre-1.0 / pre-adoption it's nearly free; after adoption you publish under the new name, ship a deprecation shim from the old (re-exporting + `DeprecationWarning`), and `yank`/`deprecate` (never delete on crates.io). **Lock the name before first publish.**
- **CLI:** `agent-sessions` plus a short `as`/`asx` alias (check PATH collisions).
- **Class:** `SessionIndex` (concept name stays even if the package is renamed).

## License

**MIT.** This isn't patent-worthy, and MIT's `AS IS` **warranty disclaimer + limitation of liability** is exactly the protection wanted if the library causes downstream damage — it shields the author from user claims (standard, generally upheld; not absolute, e.g. gross negligence; not legal advice). MIT lacks an explicit patent grant (Apache-2.0's main addition), judged unnecessary here. Keeping the library permissive maximizes adoption — commercial value lives in products built on top.

---

## Scalability

Target: one developer, local machine, ≤100K sessions, ≤1M indexed chunks, 384-dim vectors. Sync is incremental by conversation id + file mtime (unchanged set → no-op). sqlite-vec exact KNN is C-fast at this scale; switch to LanceDB ANN beyond a few million chunks.

---

## Acceptance tests

Given/When/Then, grouped. These are the library's done-definition (Dreamer-agnostic).

### Discovery & sync
- **AT-SYNC-1** Given Claude/Codex/Cursor/OpenCode data on disk, when `sync()` runs, then each present harness is indexed; absent harnesses are skipped without error.
- **AT-SYNC-2 (incremental)** Given an already-synced set, when `sync()` re-runs with no changes, then `conversations_indexed == 0` and `chunks_added == 0`.
- **AT-SYNC-3 (incremental delta)** Given one new session file, when `sync()` re-runs, then only that conversation is parsed and indexed.
- **AT-SYNC-4 (failure isolation)** Given one connector raises during parse, when `sync()` runs, then other harnesses still index and `SyncReport.harness_errors` names the failed one.

### Connectors
- **AT-CONN-1** Given a real fixture per harness, when parsed, then `Conversation` fields match the provenance table (cwd, model, harness_version, git_branch, repo_url, title).
- **AT-CONN-2 (model granularity)** Given a Codex session whose model changes mid-session, when parsed, then per-message `model` reflects the active `turn_context`.
- **AT-CONN-3 (system-injection filtering)** Given injected context (AGENTS.md, env), when parsed, then those records are not surfaced as user messages.
- **AT-CONN-4 (register)** Given a third-party `HarnessConnector`, when `register()`ed and `sync()` runs, then its sessions index without modifying library code.

### Search
- **AT-SEARCH-1 (search/hybrid)** Given a query where the best semantic and best keyword hits differ, when `search()`, then RRF fusion ranks both above unrelated results.
- **AT-SEARCH-2 (grep/keyword)** Given a literal phrase in one message, when `grep("<phrase>")`, then that locator is returned and non-matches excluded.
- **AT-SEARCH-3 (similar/vector)** Given an indexed corpus, when `similar("read before editing")`, then a semantically relevant message ranks top across harnesses.
- **AT-SEARCH-4 (filters)** Given mixed harnesses, when `search(..., harness="claude")`, then only Claude locators are returned.

### List / pagination
- **AT-LS-1 (time filter)** Given sessions across dates, when `list_conversations(since=A, until=B)`, then only sessions with `started_at ∈ [A,B]` are returned.
- **AT-LS-2 (keyset paging)** Given N > limit sessions, when paging with returned `next_cursor`, then pages are non-overlapping, complete, and stable when a new session is inserted mid-paging.

### Read / locators
- **AT-READ-1** Given a locator from a hit, when `read(locator)`, then the exact message text is returned by lazily reading the source file.
- **AT-READ-2 (broken locator)** Given a deleted/compacted source file, when `read(locator)`, then a typed "source unavailable" error is raised (not a crash), locator preserved.

### Privacy invariant
- **AT-PRIV-1** Given any indexed corpus, when the on-disk index is scanned, then **no message text appears** — only vectors, locators, and filterable scalars.
- **AT-PRIV-2** Given the default config, when indexing runs, then no network call is made (embeddings are local).

### Embedder lifecycle
- **AT-EMB-1 (mismatch)** Given an index built with embedder A, when opened with embedder B (different id/dim), then an `EmbedderMismatch` error is raised prompting a re-embed (no silent mixing of spaces).
- **AT-EMB-2 (pluggable / no runtime)** Given an API embedder configured, when indexing runs, then no ONNX runtime is loaded or fetched.

### Storage portability
- **AT-STORE-1 (single file)** Given the sqlite-vec backend, when indexing completes, then the entire index is one file that can be copied to another machine and queried unchanged.
- **AT-STORE-2 (backend swap)** Given the same corpus, when indexed with `db_backend="lancedb"`, then top hits are equivalent to sqlite-vec within recall tolerance.

### Config & harness control
- **AT-CFG-1** Given `[harnesses].enabled = ["claude"]`, when `sync()` runs, then only Claude is indexed though others are present.
- **AT-CFG-2** Given `[harnesses.cursor].enabled = false`, when `sync()` runs, then Cursor is skipped and reported disabled by `harnesses()`.

### Distribution & provisioning
- **AT-DIST-1 (light install)** Given a fresh install of the binary, when no command has run yet, then no runtime or model weights have been downloaded.
- **AT-PROV-1 (lazy first run)** Given a fresh install, when the first local-embedder `sync`/`search` runs, then runtime + weights are fetched once into the shared cache and reused on subsequent runs.
- **AT-PROV-2 (explicit setup)** Given `agent-sessions setup` in an offline-prep step, when later commands run with no network, then they succeed using the pre-fetched cache.
- **AT-PROV-3 (shared runtime)** Given both the CLI and the Tauri app installed, when each first uses local embedding, then they load the **same** cached onnxruntime, not separate copies.

### Background sync
- **AT-BG-1** Given `schedule enable --interval 30m` on macOS/Linux/Windows, when run, then the platform-native scheduler entry is created and `schedule disable` removes it.
- **AT-BG-2 (watch)** Given `sync --watch`, when a new session appears, then it is indexed within one interval without restarting the process.

### Discoverability
- **AT-DISC-1** Given `skill install`, when run, then `SKILL.md` is linked into the configured agent skill directories.
- **AT-DISC-2** Given the published package, when `llms.txt` is fetched, then it conforms to the llms.txt spec and links the quickstart, CLI reference, and connector guide.
