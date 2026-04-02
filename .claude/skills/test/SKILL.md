---
name: test
description: Use this skill when writing or running tests in OmniParser. Supplements the global test skill with project-specific environment, components, and mock patterns.
user-invocable: false
allowed-tools: [Read, Grep, Glob, Bash, Edit, Write]
---

## Environment

Always use the `omni` conda environment:
```bash
conda run -n omni pytest omnitool/gradio/tests/
conda run -n omni pytest omnitool/gradio/tests/test_file.py::test_name
conda run -n omni pytest omnitool/gradio/tests/ -m "not slow"
```

Never run bare `pytest` or `python`. If import errors appear, verify the package is installed: `pip install -e .`.

Mark slow or integration tests with `@pytest.mark.slow`.

## Key Components to Test

- `BaseAgent` (`core/agents/base.py`): `_capture_screen()`, `execute_tool_calls()`, `_compact_screen_elements()`, cost/token tracking
- `VLMAgent`: Plan→Reflect loop, `_tc_history` management, image eviction, `read_field` tool, screen-unchanged detection
- `ReActAgent`: loop detection (same `(tool_name, args)` ≥3 in last 5 actions), compaction every `COMPACTION_INTERVAL` steps, image eviction
- `AnthropicAgent`: Anthropic SDK tool calling with OmniParser grounding
- `GroundingStrategy` ABC: `OmniParserGrounding` (box_id references) and `GTA1Grounding` (natural-language target)
- Event protocol: `action_result` uses `output`/`error`; `error` uses `message`
- Tool schemas: `OMNIPARSER_COMPUTER_TOOLS`, `GTA1_COMPUTER_TOOLS`, `FINISH_TOOL`

## Mocking Patterns

- **LLM clients**: Mock `BaseLLMClient` subclasses to return `(response_text, metadata)`. `metadata` must include `tool_calls` and `assistant_message`.
- **External servers**: Mock all HTTP to OmniParser (port 8000), GTA1 (port 8002), Windows host (port 5000) — use `respx` for async HTTP.
- **Grounding**: Mock `GroundingStrategy.ground()` to return controlled `(som_image, elements)` or `(raw_image, [])`.

## Lint After Writing Tests

```bash
conda run -n omni ruff check omnitool/gradio/
```
