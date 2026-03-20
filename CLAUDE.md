# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
conda run -n omni python -m omnitool.gradio.ui.gradio.app
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

`ui/gradio/app.py` → `core/agents/factory.py:create_agent()` → agent `.run()` generator → Gradio streams each yielded event to the UI.

Every agent subclasses `BaseAgent` (`core/agents/base.py`). The base class owns `_capture_screen()`, `execute_tool_calls()`, `_compact_screen_elements()`, and all cost/token tracking. Subclasses implement three abstract hooks: `_format_messages()`, `_parse_tool_calls()`, `_get_system_prompt()`.

**Agent types:**
- `OmniAgent` — SOM box IDs, JSON response parsing, Plan→Reflect loop
- `GTAAgent` — natural-language grounding via GTA1 server, Plan→Reflect loop
- `AnthropicAgent` — Claude computer-use API with native tool calling
- `ReActAgent` — single-LLM native tool calling with pluggable `GroundingStrategy`

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

### LLM clients

All clients in `clients/llm/` inherit `BaseLLMClient` and return `(response_text, metadata)`. The `metadata` dict must include `tool_calls` and `assistant_message` when tools are used — these are required for history reconstruction. `parallel_tool_calls=False` is set in `azure.py` and `openai.py` when tools are passed.

### Event protocol

`agent.run()` is a generator. The UI (`app.py`) consumes these event types:

| `type` | Key fields | UI action |
|--------|-----------|-----------|
| `status` | `message` | Update status bar |
| `step` | `step_num` | Update status bar |
| `parsed_screen` | `som_image_base64` (empty for GTA1), `raw_image_base64`, `screen_info` | Show screenshot in chat |
| `thinking` | `response_text` | Show LLM reasoning |
| `action_result` | `tool`, `output` or `error` | Show action in chat |
| `plan` | `plan_text`, `checklist` | Show plan block |
| `complete` | `message`, `success`, `total_steps`, `total_tokens`, `total_cost` | End stream |
| `error` | `message` | End stream with error |

**Critical:** new agents must use `output`/`error` (not `result`) in `action_result`, and `message` (not `error`) in the `error` event to match `format_action_result()` in `ui/gradio/components/formatters.py`.

### Configuration & prompts

All prompts live in `config/prompts.py` and are exported through `config/__init__.py`. Builder functions (`build_vlm_system_prompt`, `build_react_system_prompt`, etc.) assemble them from `PLATFORM_PROMPTS`. Dynamic screen content is injected in **user messages** (not the system prompt) to keep the system prompt static and cacheable.

Settings priority: CLI args > YAML config file > environment variables. API keys come from environment only (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `DASHSCOPE_API_KEY`, `AZURE_OPENAI_ENDPOINT`).

### Tool schemas

`core/tools/schemas.py` defines OpenAI function schemas for `ReActAgent`. Three groups: `OMNIPARSER_COMPUTER_TOOLS` (box_id), `GTA1_COMPUTER_TOOLS` (target), and `FINISH_TOOL`. The `finish(success, summary, fields)` tool is the only exit path from the ReAct loop — there are no checklist management tools.
