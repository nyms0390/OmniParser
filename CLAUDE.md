# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

After implementing a plan, always:
1. Review the code with `/improve`
2. Run the tests with `/test`

## Environment & Setup

The active conda environment is `omni`. Always run Python commands with it:
```bash
conda run -n omni python ...
conda run -n omni pytest ...
```

Install the package in editable mode so absolute imports work:
```bash
pip install -e .
```

## Commands

**Run all tests:**
```bash
conda run -n omni pytest omnitool/gradio/tests/
```

**Run a single test file:**
```bash
conda run -n omni pytest omnitool/gradio/tests/test_agentic_loop.py
```

**Run a single test:**
```bash
conda run -n omni pytest omnitool/gradio/tests/test_agentic_loop.py::test_checklist_from_llm_json
```

**Skip slow tests:**
```bash
conda run -n omni pytest omnitool/gradio/tests/ -m "not slow"
```

**Lint (ruff):**
```bash
conda run -n omni ruff check omnitool/gradio/
```

**Start the Gradio UI:**
```bash
conda run -n omni python -m omnitool.gradio.ui.app
```

**Start the OmniParser server** (requires model weights in `weights/`):
```bash
conda run -n omni python -m omnitool.omniparserserver \
    --som_model_path weights/icon_detect/model.pt \
    --caption_model_name florence2 \
    --caption_model_path weights/icon_caption_florence
```

## System Architecture

The project has three layers that run as separate processes:

```
[util/]                    Core parser — YOLO detection, OCR, captioning
    ↓ HTTP (port 8000)
[omnitool/omniparserserver/]  FastAPI server wrapping util/
    ↓ HTTP
[omnitool/gradio/]         Agent UI & orchestration (main working area)
```

The Windows host (screenshots, mouse/keyboard) runs separately at port 5000. GTA1 grounding server runs at port 8002.

## omnitool/gradio/ Internals

### Agent execution flow

`ui/app.py` → `ui/callbacks.py:on_submit()` → `core/agents/factory.py:create_agent()` → agent `.run()` generator → Gradio streams each yielded event to the UI.

Every agent subclasses `BaseAgent` (`core/agents/base.py`). The base class owns `_capture_screen()`, `execute_tool_calls()`, `_compact_screen_elements()`, cost/token tracking, and two shared generator helpers: `_run_init()` (checklist generation + plan event, runs once before the loop) and `_run_reflect_step()` (ledger evaluation + hint injection, per step in ORCHESTRATED/TASK mode). It also handles `_handle_read_field()` and `_handle_focus_region()` inline for those tools. Subclasses implement `run()` with their own loop.

**Agent types:**
- `VLMAgent` — Plan→Reflect agent with native tool calling and pluggable `GroundingStrategy`
- `AnthropicAgent` — Claude computer-use API with Anthropic SDK tool calling; uses OmniParser for screen parsing
- `ReActAgent` — single-LLM native tool calling with pluggable `GroundingStrategy`; no Plan→Reflect loop

### VLMAgent specifics

`VLMAgent` uses the Plan→Reflect loop (via `_run_init` / `_run_reflect_step`) with native tool calling. Its design:
- **Own tool-calling history** — maintains `_tc_history` separately from `state.chat`; seeded with the task on the first step
- **Image eviction** — old images evicted from `_tc_history` on every step; history trimmed to 32 messages max (task seed + last 31)
- **Grounding** injected at construction via `GroundingStrategy` ABC; tool list and system prompt hint adapt to the active strategy
- **read_field tool** — handled inline (no external dispatch); result injected as `screen_reading` event and recorded in `working_memory.facts`
- **focus_region tool** — handled inline; crops and returns a zoomed screenshot region
- **Screen-unchanged detection** — if screen hash is identical before and after action, a warning hint is injected into `_tc_history`

### ReActAgent specifics

`ReActAgent` bypasses the Plan→Reflect loop entirely. Its design:
- **No separate planning call** — LLM outlines a plan on turn 1 via the system prompt
- **One tool call per turn** — enforced by `parallel_tool_calls=False`
- **History management** — images evicted from all but the last user message every step; harness-triggered compaction every `COMPACTION_INTERVAL` (default 8) steps asks the LLM to summarise progress (accomplished / failed attempts / current state / remaining)
- **Loop detection** — same `(tool_name, args)` ≥3 times in last 5 actions → stuck hint injected
- **Grounding** injected at construction via `GroundingStrategy` ABC (`core/agents/grounding.py`)

### GroundingStrategy

Two implementations in `core/agents/grounding.py`:
- `OmniParserGrounding` — calls OmniParser server, returns SOM-annotated image + element list; LLM references elements by integer `box_id` (0-based index into `elements` list)
- `GTA1Grounding` — passes raw screenshot to LLM; LLM uses natural-language `target`; GTA1 server resolves target → pixel coordinates

Both `VLMAgent` and `ReActAgent` accept either strategy via the `grounding` parameter (`"omniparser"` or `"gta1"`).

### Services

Business logic extracted from the UI lives in `services/`:
- `services/state.py` — `AppState`, `SessionState`, `ChatHistory`
- `services/auth.py` — `AuthProvider`, `get_api_key`, `validate_api_key`
- `services/file_handler.py` — `FileHandler` (run folder management)

### LLM clients

All clients in `clients/llm/` inherit `BaseLLMClient` and return `(response_text, metadata)`. The `metadata` dict must include `tool_calls` and `assistant_message` when tools are used — these are required for history reconstruction. `parallel_tool_calls=False` is set in `azure.py` and `openai.py` when tools are passed. `openai_responses.py` implements the OpenAI Responses API (`/v1/responses` endpoint) for models that use `api_mode="responses"`.

### External clients

External service clients live in `clients/external/`:
- `omniparser.py` — OmniParser server (port 8000)
- `gta1.py` — GTA1 grounding server (port 8002)
- `windows_host.py` — Windows host for screenshots and mouse/keyboard (port 5000)
- `paddleocr.py` — PaddleOCR service

### Event protocol

`agent.run()` is a generator. The UI (`callbacks.py`) consumes these event types:

| `type` | Key fields | UI action |
|--------|-----------|-----------|
| `status` | `message` | Update status bar |
| `step` | `step_num` | Update status bar |
| `progress` | `step`, `tokens_total`, `cost_total` | Update status bar |
| `parsed_screen` | `som_image_base64` (empty for GTA1), `raw_image_base64`, `screen_info` | Show screenshot in chat |
| `grounding` | `events` (list of `{instruction, coordinate, annotated_image_b64, success}`) | Show GTA1 grounding results |
| `thinking` | `response_text` | Show LLM reasoning |
| `action_result` | `tool`, `output` or `error`, optional `base64_image` | Show action in chat |
| `screen_reading` | `fields` | Show mid-loop field reading |
| `plan` | `plan_text`, `checklist` | Show plan block |
| `ledger` | `ledger_text`, `checklist` | Show ledger reflection |
| `assistant_reply` | `message` | Show conversational LLM response (no tool calls) |
| `extraction_result` | `fields`, `collected_facts` | Show final extracted fields |
| `complete` | `message`, `success`, `total_steps`, `total_tokens`, `total_cost` | End stream |
| `error` | `message` | End stream with error |

**Critical:** new agents must use `output`/`error` (not `result`) in `action_result`, and `message` (not `error`) in the `error` event to match `format_action_result()` in `ui/components/formatters.py`.

### Configuration & prompts

All prompts live in `config/prompts.py` and are exported through `config/__init__.py`. Builder functions (`build_vlm_tool_system_prompt`, `build_react_system_prompt`, `build_anthropic_system_prompt`) assemble them from `PLATFORM_PROMPTS`. Dynamic screen content is injected in **user messages** (not the system prompt) to keep the system prompt static and cacheable.

Settings priority: CLI args > YAML config file > environment variables. API keys come from environment only (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `DASHSCOPE_API_KEY`, `AZURE_OPENAI_ENDPOINT`).

### Tool schemas

`core/tools/schemas.py` defines OpenAI function schemas for `VLMAgent` and `ReActAgent`. Tool groups:
- `OMNIPARSER_COMPUTER_TOOLS` — computer actions using integer `box_id`
- `GTA1_COMPUTER_TOOLS` — computer actions using natural-language `target`
- `READ_FIELD_TOOL` — reads a field value from the screen via clipboard
- `FOCUS_TOOL` — crops a screen region for closer inspection (`focus_region`)
- `FINISH_TOOL` — `finish(success, summary, fields)`; the only exit path from the agent loop
