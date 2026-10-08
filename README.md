# Marcus

Marcus is a personal manager agent: one consistent conversational identity that
can remember context, use safe tools, and delegate difficult work to specialist
agents.

This repository currently contains the first conversational foundation:

- an Obsidian-compatible Markdown memory vault;
- explicit `remember`, `recall`, and recoverable `forget` operations;
- a permission policy for read-only, reversible, external, and destructive actions;
- OpenRouter conversation using `openai/gpt-6-luna` by default;
- typed intent and subsystem gating with `typesafe/jev-1.13`;
- asynchronous, policy-gated memory proposals with versioned corrections;
- a versioned communication-style profile that can adapt to repeated patterns;
- structured JSONL auditing with a separate live trace;
- streamed text responses with time-to-first-output measurement;
- non-blocking streamed text-to-speech through the macOS system voice;
- a live localhost trace dashboard;
- a shared local gateway with saved sessions, durable job status, and cancellation;
- daily Markdown fact files with a rebuildable SQLite full-text memory index;
- policy-enforced action capabilities and on-demand workflow skills;
- an optional sandboxed Codex CLI specialist;
- replaceable brain/transport contracts and compatible chat endpoints;
- a local `doctor` command for configuration checks;
- a delegated data agent that owns PDF/Markdown ingestion and its typed tool;
- hybrid keyword and semantic retrieval using `qwen/qwen3-embedding-8b`;
- a terminal shell with a single document dependency (`pypdf`).

## Project structure

The source tree mirrors the agent workflow. Start at `cli.py` for the interface,
then `runtime/service.py` for composition and the shared turn lifecycle.

```text
marcus/
├── cli.py                    # Terminal client and admin commands
├── __main__.py               # `python -m marcus` entry point
│
├── runtime/
│   ├── service.py            # Composition, foreground lane, background jobs
│   ├── state.py              # SQLite sessions, transcripts, job status
│   ├── gateway.py            # Local authenticated HTTP API + event stream
│   └── client.py             # Streaming gateway terminal client
│
├── agents/
│   ├── manager.py            # Marcus: prompt assembly and generative reply
│   ├── data.py               # Data specialist and ingestion delegation boundary
│   ├── curator.py            # Asynchronous memory/style reflection agent
│   ├── coding.py             # Optional sandboxed Codex CLI specialist
│   └── registry.py           # Explicit specialist registration
│
├── memory/
│   ├── store.py              # Daily Markdown facts + rebuildable SQLite index
│   ├── curation.py           # Deterministic acceptance/rejection of memory proposals
│   └── style.py              # Versioned adaptive communication profile
│
├── knowledge/
│   ├── ingest.py             # PDF/Markdown extraction and chunking
│   ├── embeddings.py         # OpenRouter embedding client
│   └── store.py              # SQLite index and hybrid retrieval
│
├── tools/
│   ├── manager.py            # Marcus's compact action/delegation tool surface
│   ├── jev.py                # Typed intent and subsystem-decision tool
│   └── registry.py           # Typed tools owned by specialist agents
│
├── observability/
│   ├── audit.py              # JSONL audit events and terminal trace formatting
│   ├── bus.py                # Versioned live events, bounded fan-out/replay
│   ├── web.py                # Local dashboard and live SSE event server
│   └── dashboard.html        # Browser trace interface
│
├── voice/
│   └── tts.py                # Sentence buffering, audio prefetch, playback
│
├── providers/
│   └── base.py               # Brain/transport protocols and endpoint settings
│
├── skills/
│   └── store.py              # Discover and load local SKILL.md instructions
│
└── core/
    ├── config.py             # Environment and model configuration
    ├── context.py            # Shared recent-history budget for chat/curation
    ├── policy.py             # Deterministic risks and action capabilities
    └── doctor.py             # Read-only configuration diagnostics
```

The important ownership rules are:

- Marcus owns conversation and delegation, not every tool.
- The data agent owns ingestion and its tool schema.
- Marcus receives one compact action tool and calls JEV only when intent is
  genuinely ambiguous and classification would change the next action.
- Models propose memory/style changes; deterministic memory code accepts them.
- Observability watches every layer without writing background details into chat.
- Runtime owns sessions, queueing, cancellation, and lifecycle; interfaces share it.
- Policy filters advertised actions and checks them again before execution.

## Current agentic workflow

Every user message goes directly to Marcus. A greeting, joke, or ordinary
self-contained question therefore needs one Luna request, not a JEV request
followed by a Luna request. There are no conversational slash commands.

```text
User
  │
  ▼
Terminal client / local HTTP gateway
  │
  ▼
Shared runtime → saved session history + bounded job queue
  │
  ▼
Marcus / Luna
  │
  ├─ answer directly ──────────────────────────► reply
  │
  └─ marcus_action (only when needed)
       │
       ├─ ask_jev ─────────────► typed intent/subsystem decision
       ├─ delegate_data ───────► Data agent ─► ingest tool
       ├─ recall/save/forget ──► Markdown memory
       ├─ search_knowledge ────► embeddings + document index
       ├─ status/end session
       ├─ find/read skill ────► local workflow instructions, loaded on demand
       └─ delegate_code ─────► optional Codex specialist + workspace sandbox
                 │
                 ▼
          Luna presents the result
                 │
                 ▼ when the foreground reply flags a durable signal
       background curator → memory/style proposal → deterministic validation

All paths ──► versioned event bus ──► live trace / connected clients
          └───────────────────────► JSONL audit trace
Completed conversations ─────────► SQLite session store
```

In practical terms:

1. Marcus interprets all requests from natural language; no slash syntax is
   exposed to the conversational or future voice interface.
2. Casual messages—including greetings, banter, and “tell me a joke”—are
   answered directly by Luna with no JEV call and no hardcoded phrase list.
3. A bounded profile of validated personal facts can be included as reference
   context. Additional memory and document contents are retrieved on demand.
   When a request needs context or an action, Marcus calls its compact typed
   tool. It may retrieve memory, search documents, or delegate file work.
   A titles-only catalog tells Marcus which ingested sources exist, but never
   injects their contents into ordinary turns.
4. JEV is one optional action inside that tool. Marcus asks it only when an
   ambiguous intent would change which subsystem or specialist should be used.
5. File ingestion is delegated to the data agent; Marcus never receives the
   data agent's lower-level ingestion schema.
6. If the foreground response flags a durable signal, the curator runs after
   the reply and proposes memory or style changes.
7. Deterministic validators—not the model—decide whether those changes persist.
8. Model calls, tools, policy decisions, delegation, retrieval, results, and
   latency enter the audit stream. Individual text deltas are live-only events;
   complete replies are saved once rather than logging every token.

This is the current architecture, not a future target. Planned capabilities are
kept separately in `ROADMAP.md`.

## Why Obsidian?

Obsidian is useful here because an Obsidian vault is just a folder of Markdown
files. Marcus can create and retrieve notes, while you can inspect, edit, link,
move, back up, or delete them without depending on Marcus or a database vendor.

Obsidian is **not** the intelligence. It is the human-visible long-term memory
layer. Marcus builds a rebuildable search index over these files while keeping
Markdown as the source of truth.

## Run it

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m marcus
```

Marcus loads `OPENROUTER_API_KEY` from the environment or a local `.env` file.
The `.env` file is ignored by Git. Override the default model with
`MARCUS_MODEL` if needed. Override the pinned decision model with
`MARCUS_DECISION_MODEL`; its default is `typesafe/jev-1.13`.

On macOS, spoken replies are enabled by default through the built-in `say`
engine. Text is still printed as it streams. Speech runs on a separate queue, so
the conversation and curator never wait for audio playback. Configure it with:

```bash
export MARCUS_VOICE=0
export MARCUS_VOICE_NAME=Samantha
export MARCUS_VOICE_RATE=195
```

Marcus begins printing as soon as the `reply` field starts arriving. Speech
waits for complete sentence boundaries, preserving times (`8:00 AM`), decimals,
URLs, and closing quotes. Colons, commas, and token boundaries do not create
utterances; punctuation-only fragments are discarded.

In default `stream` mode, two workers prepare upcoming audio with `say -o` while
one worker plays it in order with `afplay`. This overlaps preparation with
playback, reducing the gaps caused by restarting synthesis for every fragment.
Sentence-sized audio still has separate prosody; slow generation or synthesis
can still cause a pause. For one continuous utterance, use:

```bash
export MARCUS_VOICE_MODE=buffered
```

Buffered mode waits for the full reply before preparing speech; text still
streams. New turns interrupt playback and invalidate previous queued audio.
Without `afplay`, Marcus falls back to direct system speech. Natural-language
goodbye finishes its queued speech; Ctrl-C stops speech immediately.

Test the connection without starting an interactive session:

```bash
.venv/bin/python -m marcus --check
```

The conversation is natural language:

```text
Remember that I prefer concise explanations.
What do you remember about how I like explanations?
Forget the memory about concise explanations.
Ingest my resume at "/absolute/path/to/resume.pdf".
What source documents do you know about?
Which models are you using?
Goodbye.
```

Developer/admin operations remain process-level CLI subcommands rather than
things Marcus must interpret. For example:

```bash
.venv/bin/python -m marcus trace --follow
.venv/bin/python -m marcus ingest /absolute/path/to/resume.pdf
.venv/bin/python -m marcus sources
```

Marcus does not receive the ingestion tool schema. It delegates file work to the
data agent, and only that specialist owns and executes the typed ingestion tool.
PDF and Markdown are supported now; scanned PDFs require a future OCR stage.

## Optional JEV decision tool

JEV uses the OpenRouter Decisions API with `typesafe/jev-1.13`. It can return a
typed intent plus independent memory, document-knowledge, and reflection gates.
It is not a mandatory pre-router: Luna can invoke it through `ask_jev` when a
request is genuinely ambiguous and the answer would affect delegation.

The decision is confidence-gated. If JEV is unavailable, Marcus uses one neutral
general-query fallback without guessing intent from hardcoded phrases. JEV never
writes the reply. This avoids paying a second network round trip for clear turns
while preserving typed routing for difficult ones.

## Live trace

The browser dashboard is the easiest way to watch Marcus work:

```bash
.venv/bin/python -m marcus trace --web
```

It opens `http://127.0.0.1:8765` and updates over a local server-sent event
stream. It shows first-output latency, model duration, total foreground time,
tool actions, retrieval, JEV, curator, and voice events. Events can be searched,
filtered by subsystem, and expanded to inspect their JSON details. The server
binds only to localhost. `marcus serve` includes this dashboard and streams
directly from the event bus; do not start a second trace server on the same port.
Standalone `trace --web` still follows an embedded runtime's log files.

The dashboard shows first readable text, audio preparation, playback queue wait,
model duration, and foreground duration separately. It supports session/turn
filters and keeps expanded details open during updates. Event buffers/replay
are bounded, so slow viewers cannot block the agent or grow memory indefinitely.
Playback duration includes time spent speaking, not just system overhead.

Open a second terminal while Marcus is running:

```bash
.venv/bin/python -m marcus trace --follow
```

The normal trace is intentionally compact. It shows requested tool actions, JEV
decisions when used, first-output latency, speech progress, retrieval summaries,
prompt manifests, delegation, responses, token usage, and per-step latency
without printing the same chunks in several events. To inspect complete prompts,
embedding inputs, retrieved chunks, and JEV answers, add `--payloads`. Use
`--verbose` only for the entire JSON envelope. Background activity never appears
in the main chat.

```bash
.venv/bin/python -m marcus trace --follow --payloads
```

Credentials stored in named authorization, token, password, secret, or API-key
fields are redacted. Because the requested trace records exact prompts, ordinary
prompt text can still contain personal or sensitive information. Audit files are
local under `data/logs/` and rotate by UTC date; protect or delete them as you
would any private log.

## Document knowledge

Ingestion preserves the original file and extracted Markdown under
`data/vault/Marcus/Sources/`, creates provenance metadata, chunks the document,
and stores its search index in `data/knowledge.sqlite`. Embeddings use
`qwen/qwen3-embedding-8b` by default and degrade to keyword-only retrieval if
the embedding request fails.

Marcus retrieves at most four relevant document chunks only when it calls
`search_knowledge`; JEV can help make that decision when needed. Strong lexical
matches are resolved locally first, while embeddings remain the fallback for
semantic matches and misspelled queries. A greeting or self-contained question
never starts an embedding request. The entire resume is not copied into every
prompt: only a small titles-only source catalog is present so Marcus knows when
it should search. The original source and extracted Markdown remain the
inspectable source of truth.

By default the vault is stored at `data/vault`. To point Marcus at an existing
Obsidian vault:

```bash
export MARCUS_VAULT_PATH="/absolute/path/to/your/vault"
.venv/bin/python -m marcus
```

Marcus writes only inside a `Marcus/` folder in the selected vault.

## How automatic memory works

Marcus can include roughly 1,000 characters of validated profile/preferences/
constraints as core reference context. Other memory is retrieved on demand;
the whole vault never enters every prompt. The foreground structured response
also includes a private `reflect` flag. Once its reply is displayed, that flag
may queue a separate background reflection call. The reflector receives a
bounded working state: recent conversation, only the memories retrieved for this
turn, knowledge source references without repeated chunk text, and current
style. The main conversation does not wait for it.

The reflector proposes structured memory changes. Deterministic code accepts
only high-confidence, durable, non-secret facts in approved categories.
Duplicates and low-confidence or high-sensitivity candidates are discarded.

Corrections are versioned: the replacement fact records `supersedes: <old-id>`,
and the old fact is exported to a recoverable note in `Marcus/Archive`.
This keeps the active context clean while preserving an inspectable history.

New facts share `Marcus/Memory/Daily/YYYY-MM-DD.md`, so each fact no longer needs
its own file. Existing individual notes remain readable and are not migrated or
deleted. `Marcus/memory.sqlite` is a rebuildable full-text index; Obsidian edits
are picked up on the next retrieval. Forgetting removes the active fact and
search entry, retaining its archive note. It does not purge old transcripts/logs.

`Marcus/USER.md` and `Marcus/MEMORY.md` are bounded generated views with source
IDs. Edit source notes to change these views. Pre-existing manually written
versions are preserved and are not automatically injected into prompts. The
index preserves kind, source turn, creation time, and correction lineage.
Invalid manually edited notes are excluded without deleting them. Documents
continue to live in the knowledge index rather than being duplicated into chat
memory.

## Adaptive communication style

`MARCUS.md` remains the stable identity and operating policy. The background
reflector can separately update `data/vault/Marcus/Behavior/style.md` after it
sees a pattern across at least three user messages with high confidence. Old
revisions are copied to `data/vault/Marcus/Behavior/Archive/`, so adaptation is
inspectable and reversible.

The learned overlay may adjust brevity, formality, slang, humor, cadence, emoji,
and formatting. Validation prevents it from changing permissions, safety rules,
tool behavior, secret handling, or memory policy. It also explicitly avoids
copying accidental spelling mistakes at the expense of clarity.

Background reflection uses another model request only when Marcus's foreground
response flags a credible memory or communication-style signal. Its latency
never delays the reply, but it still has API cost when invoked. Queue wait,
request duration, proposals, and final decisions are visible in the trace.

## Gateway, sessions, and jobs

Keep the runtime in one terminal and connect chat clients to it:

```bash
.venv/bin/python -m marcus serve
# In another terminal:
.venv/bin/python -m marcus --connect http://127.0.0.1:8765 --session personal
```

`serve` runs the gateway and trace dashboard together. It continues when a
connected chat client exits, but is a foreground process, not an installed OS
startup service. `--no-open` avoids opening a browser. The existing embedded
mode still works with `.venv/bin/python -m marcus`.

The default session resumes recent saved history on the next run. `--session
<name>` creates a separate history. Histories are isolated; personal memory and
documents remain shared for this one-user agent. Only the last six completed
turns, additionally bounded by `MARCUS_HISTORY_CHARS` (default 12,000), enter the
prompt. Successful transcripts live in `data/runtime.sqlite`; failed/cancelled
partial replies do not enter history.

Gateway turns run as jobs in a serialized foreground lane; voice and reflection
have independent workers. There can be at most 32 pending/running turns. After a
restart, unfinished jobs become `interrupted` and are never silently rerun,
because tools may already have changed something. Cancellation prevents later
model/tool steps and stops speech, but cannot undo completed actions. A network
request or ingestion in progress may need to return before cancellation finishes.
This is not a resumable arbitrary workflow engine yet.

```bash
.venv/bin/python -m marcus sessions
.venv/bin/python -m marcus jobs
.venv/bin/python -m marcus doctor
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `GET /events` | SSE trace with bounded replay |
| `POST /api/turns` | Submit `{ "text": "…", "session_id": "personal" }` |
| `GET /api/jobs` / `GET /api/jobs/<id>` | Job status and final result |
| `POST /api/jobs/<id>/cancel` | Cooperative cancellation; JSON body `{}` |
| `GET /api/sessions` | Saved session list |
| `POST /api/speech/interrupt` | Stop speech; JSON body `{}` |

All `/api/` requests require `Authorization: Bearer <token>`. A token is created
in `data/gateway-token` with owner-only creation permissions; the terminal client
reads it automatically. `MARCUS_GATEWAY_TOKEN` overrides it. The gateway binds
only to localhost, checks Host/Origin headers, and rejects cross-origin browser
requests. The trace page is locally accessible without a token. It contains
personal prompts; do not expose it through a public proxy.

One chat runtime owns a data directory at a time using a POSIX process lock;
the gateway currently targets macOS/Linux. Connect to an existing owner using
`--connect`. Configurable storage locations are `MARCUS_RUNTIME_DB`,
`MARCUS_LOG_DIR`, `MARCUS_KNOWLEDGE_DB`, and `MARCUS_VAULT_PATH`.

## Capabilities, skills, and specialists

Python filters advertised actions by enabled capabilities and checks them again
before execution. Disable selected actions with a comma-separated list:

```bash
export MARCUS_DENY_ACTIONS="save_memory,forget_memory,delegate_code"
```

The data agent owns ingestion. In conversation, it uses the current user's
original request, preventing a model-supplied delegation from substituting a
different path. Imports accept PDF/Markdown up to 25 MiB;
`MARCUS_INGEST_MAX_BYTES` changes that limit. Optional `MARCUS_INGEST_ROOTS` is an
OS-path-separator-delimited allowlist checked after resolving symlinks. General
shell/computer tools are not exposed to the manager. Save/forget intent is still
interpreted by the model; capability checks cannot eliminate prompt injection.

Local skills live in `skills/<name>/SKILL.md`. Marcus sees a bounded description
catalog; `find_skill` and `read_skill` load relevant instructions on demand.
The loader limits instruction size, rejects traversal/escaping symlinks, and
traces the loaded revision. Skills cannot add tools or override policy.
`MARCUS_SKILL_DIR` selects a different directory. The first bundled workflow
describes source-based ingestion and retrieval. `marcus/skills/` contains its
loader; the top-level `skills/` directory contains actual workflow instructions.

The coding specialist is disabled by default. Enable it with an installed and
authenticated Codex CLI and a specific existing project directory:

```bash
export MARCUS_CODEX_ENABLED=1
export MARCUS_CODEX_WORKSPACE="/absolute/path/to/a/project"
```

Ask Marcus naturally to implement something there. The specialist receives the
original request, invokes `codex exec --json` with a `workspace-write` sandbox
and network disabled, and returns the final public message. It removes Marcus/
OpenRouter credentials from its environment, ignores user CLI configuration,
and uses no sandbox-bypass flags. Existing Codex authentication is used. Public
action events enter the trace; private reasoning payloads are excluded. Cancelling
terminates the process group and preserves existing edits. The default time
budget is 180 seconds (`MARCUS_CODEX_TIMEOUT`). A successful exit is not independent
verification of the edits. [Official Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)

## Model providers and bounded context

The `Brain` and `ChatTransport` protocols allow replacing the foreground model
without changing sessions, memory, tools, voice, or tracing. Defaults remain
`openai/gpt-6-luna`, `typesafe/jev-1.13`, and `qwen/qwen3-embedding-8b`.
Set `MARCUS_CHAT_API_BASE` to a compatible `/v1` base URL and
`MARCUS_CHAT_API_KEY` to its key. Local servers may omit the key:

```bash
export MARCUS_CHAT_API_BASE="http://127.0.0.1:1234/v1"
export MARCUS_MODEL="your-local-model-name"
```

The server/model must support streaming, structured JSON responses, and tools.
OpenRouter-specific parameters/credentials are not sent to another endpoint.
Reflection uses the same chat provider; `MARCUS_CURATOR_MODEL` can select a
different model. JEV and embeddings still use their separate OpenRouter clients.
For local-only operation, omit the OpenRouter key or set `MARCUS_DECISIONS=0`
and `MARCUS_EMBEDDINGS=0`. `MARCUS_REFLECTION=0` disables background reflection.

Profile, catalogs, history, and retrieved context are bounded. The same recent-
history budget applies to chat and reflection. Oversized tool responses are
returned as a limited preview with an instruction to refine the query. The manager allows
up to six model/tool rounds. There is no automatic retry after partial streaming
or tool execution, avoiding accidental duplicate actions. Automatic provider
failover and semantic memory search remain future work.

## OpenClaw inspiration and remaining work

We adopted a shared runtime/gateway, explicit capability boundaries, local memory
layers, and progressive skill discovery. Sources:
[gateway architecture](https://docs.openclaw.ai/concepts/architecture),
[memory](https://docs.openclaw.ai/concepts/memory),
[skills](https://docs.openclaw.ai/tools/skills), and
[security](https://docs.openclaw.ai/gateway/security).
Messaging plugins, scheduling, microphone input, cross-device nodes, and robot
control remain future work. This implementation keeps the current architecture
small enough to inspect and avoids advertising those capabilities as complete.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Design rules

1. Memory belongs to the user, not the model provider.
2. Models propose actions; typed tools execute them.
3. Delegated agents receive the minimum context and permissions required.
4. External and destructive actions require confirmation.
5. Every meaningful action should eventually have an audit record.

See [ROADMAP.md](ROADMAP.md) for the next stages.
