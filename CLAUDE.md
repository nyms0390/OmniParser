# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

At the start of every conversation, run `/karpathy-guidelines`.

After implementing a plan, always:
1. Review the code with `/improve`
2. Run the tests with `/test`

## Environment & Setup

Active conda env: `omni`. Always prefix Python commands:
```bash
conda run -n omni python ...
conda run -n omni pytest ...
```

Install in editable mode so absolute imports work: `pip install -e .`

## Commands

```bash
# Tests
conda run -n omni pytest omnitool/gradio/tests/

# Lint
conda run -n omni ruff check omnitool/gradio/

# Start Gradio UI
conda run -n omni python -m omnitool.gradio.ui.app

# Start OmniParser server (requires weights/)
conda run -n omni python -m omnitool.omniparserserver \
    --som_model_path weights/icon_detect/model.pt \
    --caption_model_name florence2 \
    --caption_model_path weights/icon_caption_florence
```

## System Architecture

Three layers as separate processes:

```
[util/]                      Core parser — YOLO detection, OCR, captioning
    ↓ HTTP (port 8000)
[omnitool/omniparserserver/] FastAPI server wrapping util/
    ↓ HTTP
[omnitool/gradio/]           Agent UI & orchestration (main working area)
```

External services: Windows host at port 5000 (screenshots, mouse/keyboard), GTA1 grounding at port 8002.

## omnitool/gradio/ Internals

### Agent execution flow

`ui/app.py` → `ui/callbacks.py:on_submit()` → `core/agents/factory.py:create_agent()` → agent `.run()` generator → Gradio streams events to UI.

Every agent subclasses `BaseAgent` (`core/agents/base.py`). Base class owns: `_capture_screen()`, `execute_tool_calls()`, cost/token tracking, `_handle_read_field()`, `_handle_focus_region()`, `_handle_mark_screenshot()`, `_set_fact()`, `_apply_template_aggregates()`.

**Agent types:**
- `ReActAgent` — only agent; one tool call per turn (`parallel_tool_calls=False`); harness-triggered compaction every 8 steps; pluggable `GroundingStrategy`

### GroundingStrategy

Two implementations in `core/agents/grounding.py`:
- `OmniParserGrounding` — SOM-annotated image + element list; LLM uses integer `box_id`
- `GTA1Grounding` — raw screenshot; LLM uses natural-language `target`; GTA1 server resolves to pixel coords

`ReActAgent` accepts `grounding="omniparser"` or `"gta1"`.

### Key modules

- `services/state.py` — `AppState`, `SessionState`, `ChatHistory`
- `services/auth.py` — `AuthProvider`, `get_api_key`, `validate_api_key`
- `services/file_handler.py` — `FileHandler` (run folder management)
- `clients/llm/` — LLM clients inheriting `BaseLLMClient`; return `(response_text, metadata)`; `metadata` must include `tool_calls` and `assistant_message` for history reconstruction
- `clients/external/` — `omniparser.py`, `gta1.py`, `windows_host.py`, `paddleocr.py`
- `core/tools/schemas.py` — tool schemas: `OMNIPARSER_COMPUTER_TOOLS`, `GTA1_COMPUTER_TOOLS`, `READ_FIELD_TOOL`, `FOCUS_TOOL`, `MARK_SCREENSHOT_TOOL`, `FINISH_TOOL`; `AUXILIARY_TOOLS` bundles the three always-on tools
- `config/prompts.py` — all prompts; dynamic screen content injected in **user messages** (not system prompt) to keep system prompt static and cacheable

### Event protocol

`agent.run()` yields dicts consumed by `callbacks.py`. Key types: `status`, `step`, `progress`, `parsed_screen`, `grounding`, `thinking`, `action_result`, `screen_reading`, `plan`, `ledger`, `assistant_reply`, `extraction_result`, `complete`, `error`.

**Critical:** `action_result` uses `output`/`error` keys (not `result`); `error` event uses `message` key (not `error`). Must match `format_action_result()` in `ui/components/formatters.py`.

### Configuration

Settings priority: CLI args > YAML config > environment variables. API keys from env only: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `DASHSCOPE_API_KEY`, `AZURE_OPENAI_ENDPOINT`.
