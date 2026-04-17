---
name: pitch
description: Pre-implementation idea review. Explore the codebase, validate the idea against real code, surface risks and gaps, give a concrete recommendation.
user-invocable: true
allowed-tools: [Read, Grep, Glob]
---

The user has described an idea or problem. Do NOT write any code. Your job is to validate the idea against the current codebase and give actionable feedback before implementation begins.

## Step 1 — Explore

Extract keywords and concepts from the idea. Use Glob and Grep to find relevant files, classes, and functions. Read the key ones. Understand:
- What already exists that relates to this idea
- Which modules and abstractions would be affected
- What patterns and contracts are in play (event protocol, ABC interfaces, LLM client metadata shape, etc.)

## Step 2 — Respond

Three sections. Be direct. Top 3 points per section maximum. Skip sections that have nothing useful to say.

### Relevant Context
What exists in the repo that directly bears on this idea — existing hooks, contracts, patterns, or prior art the user should know about. Ground the feedback in specific files and line references.

### Risks & Gaps
Concrete problems: wrong assumptions, missing pieces, violated interface contracts, unhandled edge cases. Cite files. Be specific (e.g. "BaseAgent._tc_history is separate from state.chat — which one does this touch?").

### Recommendation
One clear directive: what to do, or what to do instead. Include the key tradeoff in one line. If there are two genuinely different paths, list both — but pick one.

## Tone

Direct. Skip what's obviously fine. If the idea is mostly sound, say so in one line and focus on what needs attention.
