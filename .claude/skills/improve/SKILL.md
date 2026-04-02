---
name: improve
description: Use this skill when reviewing or improving OmniParser code. Supplements the global improve skill with project-specific conventions.
user-invocable: false
allowed-tools: [Read, Grep, Glob, Bash, Edit]
---

## OmniParser-Specific Conventions

When reviewing code in this project, flag violations of the following as **Important** or **Critical**:

- **Linting**: Project uses `ruff` — flag anything ruff would catch
- **Type hints**: Expected throughout all Python code
- **Event protocol**: `action_result` events must use `output`/`error` keys (not `result`); `error` events must use `message` key (not `error`)
- **System prompts**: Must be static and cacheable — dynamic/screen content goes in user messages, not the system prompt
- **`parallel_tool_calls=False`**: Required in `azure.py` and `openai.py` when tools are passed (enforced for `ReActAgent`)
- **LLM client metadata**: `metadata` dict from all `BaseLLMClient` subclasses must include `tool_calls` and `assistant_message` keys
- **Abstract base classes**: Agents subclass `BaseAgent`; grounding strategies subclass `GroundingStrategy`; LLM clients subclass `BaseLLMClient`
- **Conda commands**: Always `conda run -n omni python ...`, never bare `python`

## Running the Linter

```bash
conda run -n omni ruff check omnitool/gradio/
```
