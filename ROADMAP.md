# Marcus roadmap

## Phase 0 — foundation (current)

- Human-readable Markdown memory
- Recoverable forgetting
- Permission/risk model
- Local terminal interface

## Phase 1 — conversational brain (in progress)

- OpenRouter brain using structured responses
- Automatic memory curation and versioned corrections
- Structured JSONL audit stream and live trace viewer
- PDF/Markdown ingestion with source provenance
- Hybrid semantic and keyword document retrieval
- JEV intent routing and memory/knowledge/reflection gates
- Initial delegated data agent for PDF/Markdown ingestion
- Provider-neutral brain/transport contracts and compatible endpoints (complete)
- Persisted transcripts and bounded session context (complete)
- Daily fact files, profile views, and rebuildable SQLite memory index (complete)
- Progressive local workflow skill discovery (complete)
- Build a small evaluation set from real Marcus requests

## Phase 2 — safe tools

- Open applications
- Search files
- Read clipboard and selected screen context
- Create notes and reminders
- Add dry-run previews, audit history, and confirmations

## Phase 3 — voice (in progress)

- Local wake word and voice-activity detection
- Streaming speech recognition
- Streaming speech synthesis (initial macOS system-voice backend complete)
- Interruption support (typed-turn interruption complete)
- Speaker verification for privileged requests

## Phase 4 — delegation

- Gateway, serialized jobs, saved status, cancellation, crash recovery (complete)
- Explicit specialist registry and optional sandboxed Codex CLI agent (complete)
- True resumable workflows and independent specialist lanes
- Research and document specialists
- Progress events, cancellation, budgets, and verification gates

## Phase 5 — embodiment

- Microphone array, speaker, display, and camera
- Separate microcontroller for motor safety
- Wheeled base with collision avoidance
- Room awareness and optional Home Assistant integration
- Physical mute and emergency-stop controls

## Target architecture

```text
terminal / gateway client / future voice and robot
        |
        v
shared runtime + persisted sessions + live event bus
        |
        v
Marcus conversation manager
        |
        v
JEV decision tool + memory + policy + job manager
        |
        +-------- data agent + typed ingestion tools
        +-------- deterministic workflows
        +-------- optional Codex CLI coding specialist
        +-------- on-demand workflow skills
        +-------- future device nodes and specialists
```
