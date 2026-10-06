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
- Add a formal provider-neutral `Brain` protocol
- Add a local/OpenAI-compatible adapter
- Keep prompts, identity, memory, and tools outside provider-specific code
- Add conversation transcripts and explicit memory extraction proposals
- Build a small evaluation set from real Marcus requests

## Phase 2 — safe tools

- Open applications
- Search files
- Read clipboard and selected screen context
- Create notes and reminders
- Add dry-run previews, audit history, and confirmations

## Phase 3 — voice

- Local wake word and voice-activity detection
- Streaming speech recognition and speech synthesis
- Interruption support
- Speaker verification for privileged requests

## Phase 4 — delegation

- Durable background job manager
- Specialist registry and typed task/result contracts
- Codex adapter for coding tasks
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
voice / text / robot
        |
        v
Marcus conversation manager
        |
        v
JEV router + memory + policy + job manager
        |
        +-------- data agent + typed ingestion tools
        +-------- deterministic workflows
        +-------- specialist agents
```
