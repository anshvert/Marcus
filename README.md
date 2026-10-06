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
- a delegated data agent that owns PDF/Markdown ingestion and its typed tool;
- hybrid keyword and semantic retrieval using `qwen/qwen3-embedding-8b`;
- a terminal shell with a single document dependency (`pypdf`).

## Why Obsidian?

Obsidian is useful here because an Obsidian vault is just a folder of Markdown
files. Marcus can create and retrieve notes, while you can inspect, edit, link,
move, back up, or delete them without depending on Marcus or a database vendor.

Obsidian is **not** the intelligence. It is the human-visible long-term memory
layer. Later, Marcus can build a search index over these files without making
the index the source of truth.

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

Test the connection without starting an interactive session:

```bash
.venv/bin/python -m marcus --check
```

Useful commands:

```text
/remember I prefer concise explanations
/recall explanations
/memories
/forget <memory-id>
/ingest /absolute/path/to/resume.pdf
/sources
/help
/quit
```

Marcus can also invoke ingestion from ordinary language:

```text
Ingest my resume at "/absolute/path/to/resume.pdf".
```

Marcus does not receive the ingestion tool schema. The router delegates explicit
file work to the data agent, and only that specialist owns and executes the
typed tool. PDF and Markdown are supported now; scanned PDFs require a future
OCR stage.

## Routing and subsystem gates

Obvious commands and greetings use deterministic rules because an exact rule is
faster and more reliable than calling a model. Other natural-language messages
go through one OpenRouter Decisions API request using `typesafe/jev-1.13`. That
single typed decision chooses the route and independently decides whether the
turn needs personal memory, document knowledge, or background reflection.

The decision is confidence-gated and falls back to conservative local rules if
JEV is unavailable. JEV never writes the reply: Luna is still used when prose or
reasoning is needed. A simple greeting uses neither model.

## Live trace

Open a second terminal while Marcus is running:

```bash
.venv/bin/python -m marcus trace --follow
```

The normal trace is intentionally compact. It shows routes and gate
probabilities, retrieval summaries, prompt manifests, delegation, responses,
token usage, and per-step latency without printing the same chunks in several
events. To inspect complete prompts, embedding inputs, retrieved chunks, and JEV
answers, add `--payloads`. Use `--verbose` only for the entire JSON envelope.
Background activity never appears in the main chat.

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

Marcus retrieves at most four relevant document chunks only when the JEV gate
judges document knowledge relevant. A greeting or self-contained question never
starts an embedding request. The entire resume is not copied into every prompt.
The original source and extracted Markdown remain the inspectable source of
truth.

By default the vault is stored at `data/vault`. To point Marcus at an existing
Obsidian vault:

```bash
export MARCUS_VAULT_PATH="/absolute/path/to/your/vault"
.venv/bin/python -m marcus
```

Marcus writes only inside a `Marcus/` folder in the selected vault.

## How automatic memory works

JEV first decides whether personal memory can materially help. If not, Marcus
does not read or inject it. Once a model-generated reply is displayed, JEV may
queue a separate background reflection call. The reflector receives a bounded
working state: recent conversation, only the memories retrieved for this turn,
knowledge source references without repeated chunk text, and current style. The
main conversation does not wait for it.

The reflector proposes structured memory changes. Deterministic code accepts
only high-confidence, durable, non-secret facts in approved categories.
Duplicates and low-confidence or high-sensitivity candidates are discarded.

Corrections are versioned rather than silently overwritten: the replacement
note records `supersedes: <old-id>`, and the old note moves to `Marcus/Archive`.
This keeps the active context clean while preserving an inspectable history.

Marcus does not copy the whole vault into the prompt. The current lexical
retriever selects up to eight relevant notes and caps their total size. A later
hybrid index can combine metadata, keyword search, embeddings, recency, and
project scope while leaving Markdown as the source of truth.

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

Background reflection uses another model request only when the JEV gate sees a
credible memory or communication-style signal. Its latency never delays the
reply, but it still has API cost when invoked. Queue wait, request duration,
proposals, and final decisions are visible in the trace.

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
