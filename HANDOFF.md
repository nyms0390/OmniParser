# Handoff Document — OmniParser / omnitool

## Goal

Replace the legacy `OmniAgent` and `GTAAgent` with a unified `VLMAgent` that uses OpenAI native function calling and a pluggable `GroundingStrategy` (OmniParser SOM or GTA1 natural-language). Deliver a clean, well-tested agent codebase on the `refactor/enterprise-architecture` branch.

## Current Progress

All planned implementation work is complete:

- **`VLMAgent`** implemented in `omnitool/gradio/core/agents/vlm_agent.py` (~610 lines)
  - Plan→Reflect loop with native tool calling (`_tc_history`)
  - Pluggable `GroundingStrategy` (OmniParserGrounding / GTA1Grounding)
  - `read_field` tool handled inline with optional clipboard correction
  - Image eviction + history trimming (max 32 messages) to avoid context overflow
  - Screen-change detection fixed (was always comparing screen to itself)
  - `complete` event includes `message` and `success` fields
- **`OmniAgent` and `GTAAgent` deleted** — factory routes both legacy names to `VLMAgent`
- **`VLM_TOOL_SYSTEM_PROMPT`** added; legacy prompts deleted
- **`READ_FIELD_TOOL`** schema added to `core/tools/schemas.py`
- **`WorkingMemory.screen_data`** field added to `base.py`
- **UI settings** updated: `VLMAgent` is default, all 3 agent names available
- **Test suite rewritten** — `test_agentic_loop.py` fully updated for `VLMAgent` (59 passing, 0 failing)
- **`/handoff` skill** installed at `~/.claude/skills/handoff/SKILL.md`

### Bug fixes applied in this session (code review of `core/agents/`)

Four bugs were found and fixed:

1. **`_reflect_done` not initialized in `__init__`** (`base.py`) — added `self._reflect_done = False` to `__init__` alongside `working_memory`. Previously a dynamic attribute only set inside `_run_reflect_step()`; any out-of-order access would raise `AttributeError`.

2. **Zero-coordinate sentinel ambiguous** (`base.py:_read_field_via_clipboard`) — changed `if not rx and not ry:` to `if rx == 0 and ry == 0:`. The old form used Python's falsy check, which would also trip on `float(0)` or other zero-equivalent values; the new form is explicit about the GTA1 sentinel.

3. **Multipart task content silently corrupts task string** (`base.py:_generate_checklist`, `react_agent.py:run`) — both sites that read `messages[0]["content"]` now call `_extract_text_content()`, a new module-level helper in `base.py` that extracts text from a list-of-blocks content (multimodal message) or returns the string as-is. Without this, if the user attached an image to their task message, `working_memory.task` would be set to a Python list, and string interpolation in every subsequent step's task reminder would produce garbage.

4. **`OmniParserClient.parse_screenshot()` docstring wrong** (`clients/external/omniparser.py`) — the docstring documented return keys `som_image_base64` / `original_screenshot_base64` but the server actually returns `labeled_screenshot_base64` / `parsed_content_list`. Updated to match reality.

## What Worked

- Overriding `run()` completely in `VLMAgent` (same pattern as `ReActAgent`) rather than hooking into `BaseAgent`'s text-parsing loop — avoided impedance mismatch entirely
- Mocking `_do_capture`, `_generate_checklist`, `_reflect`, and `execute_tool_calls` independently in tests — kept each test focused without cascading mock complexity
- `screen_before` snapshot + temporary swap before `_verify_step` — clean fix for the screen-comparison bug without changing the base class interface
- LEGACY_ rename-then-delete approach for old prompts — safe transition without breaking anything mid-refactor

## What Didn't Work

- Putting skill files as flat `~/.claude/skills/handoff.md` — Claude Code requires `~/.claude/skills/<name>/SKILL.md` (subdirectory + uppercase filename)
- `app_state.run_folder` — doesn't exist; correct path is `app_state.session.run_folder`
- Pre-existing `test_core_components.py` has an unrelated `ImportError` on `get_model_config` — excluded from test runs with `--ignore`

## Next Steps

1. **Manual smoke test** with real OmniParser weights to verify end-to-end VLMAgent execution
2. **GTA1Grounding path** — verify `GTA1Client` clipboard correction works in a live session (unit tests mock it)
3. **`test_core_components.py`** — fix the pre-existing `ImportError: cannot import name 'get_model_config'` so all tests can run without `--ignore`
4. **Consider ReActAgent compaction parity** — VLMAgent trims history to 32 messages (simple slice); ReActAgent uses LLM-summarised compaction; may want to unify
5. **Open a PR** from `refactor/enterprise-architecture` → `master` when ready (user has decided NOT to push yet — branch is local-only by choice)
6. **Trajectory replay feature** — see design discussion below

---

## Design Discussion: Trajectory Replay

### Idea
After a successful task run, record the tool call trajectory `[{tool, args}, ...]`. On future runs, replay it directly (no LLM calls) for speed/cost savings. Fall back to full LLM if replay diverges.

### Key findings from analysis

**Why raw replay is brittle:**
- `box_id` is a 0-based index into OmniParser's detected element list for *that specific screenshot* — it has no meaning across runs. Even minor UI state changes shift all indices.
- Replay fails silently: tool calls (click, type) almost never return errors even when they hit the wrong target.

**Why "trajectory as hint" is worse:**
- Feeding the recorded trajectory as LLM context causes anchoring/hallucination. The model rationalizes following the old trajectory even when the UI has changed. Fails silently with false confidence — worse than hard failure.

**The core unsolved problem: verification**
Tool call success cannot be detected from return values alone. Verification must be vision-based.

### Recommended design

```
1. Replay trajectory step by step
2. After each step (or at milestone checkpoints), verify screen state
3. If divergence detected → abort replay, hand off to fresh LLM run (no hint)
```

**Verification approaches (in order of preference for this codebase):**

| Approach | How | Robustness |
|----------|-----|-----------|
| OmniParser element list comparison | Record expected elements after each step; compare at replay time | Best fit — already running |
| Milestone-only checkpoints | Only verify at semantically meaningful steps (dialog close, page nav) | Practical, low overhead |
| Perceptual hash / CLIP embedding | Record screenshot embedding; check cosine distance | Good for gross divergence |
| LLM-as-verifier | Send before+after images with binary prompt — cheap (~50-100 tokens) | Most reliable, some cost |

**Practical recommendation:** OmniParser element comparison at milestone checkpoints (not every step). On divergence, fall back to a fresh LLM run starting from the divergence point — passing only current screen state, not the old trajectory.

**Natural implementation point:** a `TrajectoryReplayAgent` or a `ReplayGrounding` strategy that wraps `OmniParserGrounding` and short-circuits LLM calls for pre-recorded steps.

## Key Files

| File | Purpose |
|------|---------|
| `omnitool/gradio/core/agents/vlm_agent.py` | Main implementation |
| `omnitool/gradio/core/agents/grounding.py` | GroundingStrategy ABC + OmniParser/GTA1 impls |
| `omnitool/gradio/core/agents/factory.py` | Agent construction, routes OmniAgent/GTAAgent → VLMAgent |
| `omnitool/gradio/core/agents/base.py` | BaseAgent, WorkingMemory, _reflect_done, _extract_text_content |
| `omnitool/gradio/core/tools/schemas.py` | Tool schemas incl. READ_FIELD_TOOL |
| `omnitool/gradio/config/prompts.py` | VLM_TOOL_SYSTEM_PROMPT, REFLECT_PROMPT |
| `omnitool/gradio/tests/test_agentic_loop.py` | Full test suite (59 tests) |
| `omnitool/gradio/clients/external/omniparser.py` | OmniParserClient (docstring corrected) |
| `CLAUDE.md` | Dev commands, architecture overview |

## Run Tests

```bash
conda run -n omni python -m pytest omnitool/gradio/tests/ --ignore=omnitool/gradio/tests/test_core_components.py -q
# Expected: 59 passed
```
