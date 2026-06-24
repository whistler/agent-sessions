# agent-sessions

Search your local coding-agent sessions — Claude, Codex, Cursor, OpenCode —
by meaning or regex. **Stores vectors + locators only. Your transcripts never
leave your machine and are never copied into the index.**

> Status: early design. See [DESIGN.md](DESIGN.md) for the full architecture.

## Install

```bash
curl -fsSL https://github.com/whistler/agent-sessions/releases/latest/download/install.sh | sh
brew install whistler/tap/agent-sessions    # mac/linux
cargo binstall agent-sessions               # cargo users, prebuilt
cargo install agent-sessions               # from source
```

## 30 seconds

```bash
agent-sessions sync                        # index all sessions (incremental, ~seconds)
agent-sessions search "always read before editing"
agent-sessions grep "use pnpm" --json      # machine-readable output
```

No API key required — embeddings run locally on first sync (model downloaded once, ~30 MB).

## Search

| verb | does |
|---|---|
| `search` | hybrid (semantic recall + keyword re-rank) — the smart default |
| `grep`   | regex / literal — exact strings |
| `similar`| pure vector nearest-neighbor |

## CLI

```
agent-sessions sync                         # discover + index new sessions (incremental)
agent-sessions search "<q>"                 # hybrid search; --harness, --project, --json
agent-sessions grep "<regex>"               # keyword across all harnesses
agent-sessions similar "<q>"               # vector nearest-neighbor
agent-sessions ls [--harness H] [--json]   # list conversations
agent-sessions show <id>                   # inspect a conversation
agent-sessions harnesses                   # list connectors + present/absent status
agent-sessions setup                       # pre-fetch model weights (for CI / offline)
agent-sessions skill install               # symlink SKILL.md into agent skill dirs
```

`--json` on every read command emits stable machine-readable output. See `SKILL.md` for
agent-oriented usage patterns, and `llms.txt` for LLM discoverability.

## Supported harnesses

Claude Code · Codex · Cursor · OpenCode. Add your own — see the connector
trait in [DESIGN.md](DESIGN.md).

## Privacy

Local-first. The index holds embeddings + pointers back into the original
session files. Text is read lazily on demand and never persisted.

## License

[MIT](LICENSE).
