# agent-sessions

Search your local coding-agent sessions — Claude, Codex, Cursor, OpenCode —
by meaning or regex. **Stores vectors + locators only. Your transcripts never
leave your machine and are never copied into the index.**

> Status: early design. See [DESIGN.md](DESIGN.md) for the full architecture.

## Install

```bash
curl -fsSL https://agent-sessions.dev/install.sh | sh   # universal one-liner
brew install <you>/tap/agent-sessions                   # mac/linux
cargo binstall agent-sessions                            # cargo users, prebuilt
```

For Dreamer / Python consumers:

```bash
pip install agent-sessions      # or: uv add agent-sessions
```

## 30 seconds

```python
from agent_sessions import SessionIndex

idx = SessionIndex()                  # zero config: auto-detect harnesses, local embedder
idx.sync()
for hit in idx.search("always read before editing"):
    print(hit.score, hit.locator)
```

No API key required — embeddings run locally.

## Search

| verb | does |
|---|---|
| `search` | hybrid (semantic recall + keyword re-rank) — the smart default |
| `grep`   | regex / literal — exact strings |
| `similar`| pure vector nearest-neighbor |

## CLI

```
agent-sessions sync                  # discover + index new sessions (incremental)
agent-sessions search "<q>"          # hybrid search; --harness, --project, --json
agent-sessions grep "<regex>"        # keyword across all harnesses
agent-sessions similar "<q>"         # vector nearest-neighbor
agent-sessions ls | show <id>        # list / inspect conversations
```

## Supported harnesses

Claude Code · Codex · Cursor · OpenCode. Add your own — see the connector
trait in [DESIGN.md](DESIGN.md).

## Privacy

Local-first. The index holds embeddings + pointers back into the original
session files. Text is read lazily on demand and never persisted.

## License

[MIT](LICENSE).
