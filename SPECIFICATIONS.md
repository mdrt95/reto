# Implementation Specifications — Banorte CV Agent

This index is the implementation sequence. Each numbered specification is independently completable and ends at a safe stop/resume boundary. Do not begin a later specification until the current one meets its completion gate.

| Order | Specification | Purpose | Safe stop when |
|---|---|---|---|
| 00 | [Foundation and profile](specs/00-foundation-and-profile.md) | Establish the repository, runtime, validated source data, and configuration | The application can load its only runtime source of truth and answer `/health` |
| 01 | [Core agent](specs/01-core-agent.md) | Implement bounded agent orchestration, typed tools, generation, grounding, and guards | The agent service produces verified answers without HTTP integration |
| 02 | [Public API and frontend](specs/02-public-api-and-frontend.md) | Add the stable first-party chat contract and static user interface | A browser can safely use the deployed-contract shape locally |
| 03 | [Quality, operations, and deployment](specs/03-quality-operations-and-deployment.md) | Add focused tests, evaluation, observability, containerization, and Render release checks | The public demo has passed its release gates |

## Global invariants

- `data/profile.json` is the only runtime claim source; `MDRT Resume.json` is reviewed authoring input.
- The public core contract is `POST /api/chat`. External protocol adapters are out of scope until selected and contract-tested.
- Profile queries and grounding are deterministic first. No embedding, chunking, vector database, durable chat memory, or write-capable tools exist in v1.
- User input never becomes system instructions. All public errors are sanitized and all logs exclude content and contact data.
- Tests remain minimal and risk-based, as defined in specification 03.

## Resume protocol

When work resumes, inspect this index, the last completed specification's completion gate, and `DECISIONS.md`. Continue with the first incomplete specification; do not reopen accepted decisions without a concrete new requirement.
