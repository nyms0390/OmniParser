---
name: pitch
description: Pre-implementation idea review. Explore the codebase, challenge assumptions, surface gaps, suggest improvements. Use before writing any code.
user-invocable: true
allowed-tools: [Read, Grep, Glob]
---

The user has described an idea. Do NOT write any code. Your job is to think critically and helpfully before implementation begins.

## Step 1 — Explore

Extract keywords and concepts from the idea. Use Glob and Grep to find relevant files, classes, and functions. Read the key ones. Understand:
- What already exists that relates to this idea
- Which modules and abstractions would be affected
- What patterns and contracts are in play (event protocol, ABC interfaces, LLM client metadata shape, etc.)

## Step 2 — Respond

Produce a short, structured response in three sections. Be direct. Top 3 points per section maximum.

### Challenges
What assumptions in the idea might be wrong or risky? What could go wrong? Cite specific files or patterns where relevant.

### Implementation Details Not Addressed
Concrete gaps the idea doesn't cover — error paths, streaming event types, interface contracts, test surface, edge cases. Be specific (e.g. "VLMAgent._tc_history is separate from state.chat — which one does this touch?").

### Suggestions
Alternatives or refinements worth considering, with a one-line tradeoff for each.

## Tone

Balanced: surface real risks and gaps, but also note what's sound about the approach. Punchy — no filler.
