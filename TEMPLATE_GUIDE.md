# Task Template Guide

A **task template** is a `.yaml` file that tells the AI agent exactly what to do, step by step. You write it once; the agent reads it and carries out the task on a real computer screen.

---

## What is YAML?

YAML is a plain-text format that uses **indentation (spaces)** to organize information. The rules are:
- Use **spaces**, never tabs.
- Indentation matters — a line indented more is "inside" the one above it.
- Lines starting with `-` are list items.
- Lines with `:` separate a label from its value.

---

## File Structure at a Glance

A template has two top-level sections: **`inputs`** (shared values available to all procedures) and **`procedures`** (the actual tasks to run).

```yaml
inputs:
  - key: employee_id
    value: 12345678
    description: "The employee's 8-digit ID number."
    format: "8 digits"
    required: true

procedures:
  - ID: 1
    description: "What this procedure does"
    inputs: [employee_id]
    outputs:
      - key: confirmation_number
        description: "The confirmation number shown after submission."
        format: string
    executions:
      - type: cua
        system: MySystem
        steps: |
          1. Do the first thing using <employee_id>.
          2. Do the second thing and record the result as {confirmation_number}.
```

---

## Section 1: `inputs` — Shared Values

Inputs are defined **once at the top of the file** and can be reused across multiple procedures. Each input is a named slot holding a value the agent will use.

```yaml
inputs:
  - key: employee_id
    value: 87654321
    description: "The employee's 8-digit ID number."
    format: "8 digits"
    required: true
  - key: report_month
    value: February
    description: "The month for which the report is filed."
    format: "Full month name"
    required: false
```

| Field | Required? | What to put here |
|---|---|---|
| `key` | **Yes** | A short name with no spaces (use underscores). This is how you reference it in steps: `<key>`. |
| `value` | No | The actual value to use. Leave blank if the user will fill it in at runtime. |
| `description` | No | Plain-English explanation of what this input is. |
| `format` | No | Describes the expected shape of the value (e.g., `8 digits`, `MM/DD/YYYY`, `string`). |
| `required` | No | `true` or `false`. Defaults to `false`. |

---

## Section 2: `procedures` — The Task Steps

Procedures are the actual instructions the agent will follow. You can have multiple procedures in one file, each with a different `ID`. When a template is loaded, all procedures appear in a dropdown and you select which one to run. If no selection is made, the first procedure is used as the default.

```yaml
procedures:
  - ID: 1
    description: "Submit a monthly expense report in Concur."
    inputs: [employee_id, report_month]
    outputs:
      - key: report_number
        description: "The report ID shown after submission."
        format: string
    executions:
      - type: cua
        system: Concur
        steps: |
          1. Open Concur and log in with <employee_id>.
          2. Navigate to Reports > Monthly Submission.
          3. Fill in the report for <report_month>.
          4. Click Submit and note the report number as {report_number}.
```

### `ID`
A whole number (1, 2, 3…). Required.

### `description`
A plain-English sentence describing what this procedure does.

### `inputs`
A list of input **key names** (from the top-level `inputs` section) that this procedure uses. Use the inline bracket format:

```yaml
inputs: [employee_id, report_month]
```

Or the expanded list format:
```yaml
inputs:
  - employee_id
  - report_month
```

> **Important:** Only list keys that are already defined under the top-level `inputs` section. Any key not found there will be silently skipped.

### `outputs`
The pieces of information the agent should capture when the procedure is done.

```yaml
outputs:
  - key: report_number
    description: "The report ID number shown after submission."
    format: string
```

| Field | Required? | What to put here |
|---|---|---|
| `key` | **Yes** | Short name with no spaces. Reference it in steps as `{key}` to signal the agent to capture that value. |
| `description` | No | Tells the agent what to look for on screen. |
| `format` | No | Expected format of the captured value. |
| `clipboard_correction` | No | `true` (default) or `false`. When `true`, the agent captures the value by tri-clicking the field to select its content, then reading the clipboard. **Before enabling this, verify that the target system's input fields support tri-click selection** (triple-click selects the full field content). Set to `false` for fields where tri-click does not select text, or where the value only needs to be visually confirmed rather than extracted. |
| `dynamic` | No | `false` (default) or `true`. When `true`, the agent captures this key once per matching step and accumulates all captured values into a list (useful for reading the same field across multiple rows). |
| `aggregate` | No | Declares an auto-computed output. **Not captured by the agent directly** — computed at finish from a dynamic field. Requires two sub-fields: `operation` (see table below) and `source` (the `key` of the dynamic output to aggregate). |

**Aggregate operations:**

| `operation` | Result type | What it does |
|---|---|---|
| `sum` | string (number) | Adds all captured values numerically. Values may include currency symbols or commas — non-numeric characters are stripped before summing. |
| `concat` | string | Joins all captured values into one string with no separator. |
| `none` | list | Returns the raw list of captured values unchanged. |
| `dedup` | list | Returns the list with duplicate values removed, preserving order. |

**Dynamic + aggregate pattern** — use this when a value appears once per row and you want a total:

```yaml
outputs:
  - key: row_fee
    description: "Fee shown on each row of the table."
    format: decimal
    dynamic: true          # captured once per row, accumulates a list
  - key: total_fee
    description: "Sum of all row fees."
    format: decimal
    aggregate:
      operation: sum
      source: row_fee      # aggregated from the dynamic field above
```

The agent captures `row_fee` on every relevant step; `total_fee` is computed automatically when the procedure finishes. Do **not** reference `{total_fee}` in steps — only reference `{row_fee}` where it should be read.

> **Key rules:** All key names must be unique across the entire template — no input and output may share the same name. Each key also holds exactly one value; do not pack multiple pieces of information into a single key (e.g., use `first_name` and `last_name`, not one `full_name` key for both).

### `executions`
Where the actual step-by-step instructions live.

| Field | What to put here |
|---|---|
| `type` | Always `cua` (Computer Use Agent — GUI automation). |
| `system` | A short name for the application being used (e.g., `Concur`, `EPA`, `Salesforce`). Informational only. |
| `steps` | Your step-by-step instructions. **Must be followed by ` |` on the same line.** |

---

## How to Write Steps

Steps are the most important part. The agent reads them as a checklist and works through them one by one.

### Numbered list (recommended)
```
1. Open Chrome and go to the login page.
2. Type <employee_id> into the username field.
3. Click the Submit button.
```

### Bullet list
```
- Open Chrome and go to the login page.
- Type <employee_id> into the username field.
- Click the Submit button.
```

**Rules:**
- Each step should describe **one action** (click, type, navigate, wait).
- Be specific: say *where* to click, *what* to type, *which* button/field/menu.
- Reference inputs with `<key_name>` — replaced with real values before the agent runs.
- Mark outputs for capture with `{key_name}` — tells the agent to record that value.
- Avoid vague language like "fill in the form." Instead: "Type `<employee_id>` into the field labeled 'Employee ID'."
- **Write explicit scroll steps.** The agent cannot reliably judge when a page needs scrolling. If a button, field, or section might be off-screen, add a dedicated step before it: `"Scroll down until the Confirm button is visible."` or `"Scroll to the bottom of the page."` Do not assume the agent will scroll on its own.
- **Describe ambiguous UI elements by their neighbors.** Some widgets (dropdowns, toggles, icon buttons) have no visible label. Identify them by position relative to a nearby labeled element: `"Click the dropdown to the right of the 'Department' label."` or `"Focus the input field below the 'Start Date' heading."` Avoid descriptions that rely solely on visual appearance (color, shape, size).

### Optional: Verification hints

After any step, add an indented `verify:` line. The agent uses this to confirm the step succeeded before moving on.

```
1. Click the "Submit" button.
   verify: A green "Submission Successful" banner appears at the top of the page and the URL changes to /confirmation.
2. Note the confirmation number as {confirmation_number}.
   verify: A 10-digit confirmation number is displayed in bold under the heading "Your submission ID".
```

Write verification hints as **concrete observations**: name the exact element, text, color, or screen state that proves success. Vague hints like "the page updates" or "a message appears" are not useful — specify *which* message and *where* it appears.

The `verify:` line must be:
- On its own line, immediately after the step it belongs to.
- Indented (at least one space more than the step).
- Starting with `verify:` (case-insensitive).

---

## Two Ways to Reference Values in Steps

| Syntax | Purpose | Example |
|---|---|---|
| `<key_name>` | **Insert** an input value into the step | `Type <employee_id> into the login field.` |
| `{key_name}` | **Capture** an output value at this step | `Record the confirmation code as {confirmation_number}.` |

Only outputs referenced with `{key_name}` in steps will appear in the final "Outputs to capture" summary sent to the agent.

---

## Complete Example

```yaml
inputs:
  - key: employee_id
    value: 12345678
    description: "8-digit employee ID."
    format: "8 digits"
    required: true
  - key: report_month
    value: March
    description: "Month being reported."
    format: "Full month name"
    required: true

procedures:
  - ID: 1
    description: "Submit a monthly expense report in the Concur system."
    inputs: [employee_id, report_month]
    outputs:
      - key: report_number
        description: "The report ID number shown after submission."
        format: string
    executions:
      - type: cua
        system: Concur
        steps: |
          1. Open the Concur website and log in with employee ID <employee_id>.
             verify: The Concur dashboard is visible.
          2. Click "Expense" in the top navigation bar.
          3. Click "Create New Report".
          4. Set the report name to "Monthly Expenses - <report_month>".
          5. Click "Submit Report".
             verify: A confirmation page appears with a report number.
          6. Note the report number as {report_number}.
             verify: A report number is visible on the confirmation page.
```

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| Using tabs instead of spaces | Always use the spacebar for indentation |
| Forgetting the `\|` after `steps:` | Must be `steps: \|` — the pipe character is required |
| Steps text at wrong indent level | Steps must be indented further than `steps:` |
| Referencing an input not defined in top-level `inputs` | Make sure every key in `procedure.inputs` exists in top-level `inputs` |
| Writing `required: True` with capital T | Use lowercase: `true` / `false` |
| Defining inputs inside the procedure (old format) | All inputs must be at the **top level**, not inside a procedure |
| Using `<key>` for outputs | Outputs use curly braces: `{key}`, not angle brackets |
| Duplicate key names across inputs/outputs | All key names must be unique across the entire template |
| Mapping multiple values to one key | Define a separate key for each distinct piece of information |
| Vague `verify:` hints like "page updates" | Name the exact element, text, or screen state: which message, what it says, where it appears |
| No scroll step before an off-screen element | Add an explicit step: "Scroll down until the Submit button is visible." |

---

## Quick Checklist Before Saving

- [ ] File starts with `inputs:` then `procedures:`
- [ ] All indentation uses spaces (not tabs)
- [ ] Every input has a `key:` field
- [ ] Every procedure has `ID:`, `description:`, and `executions:`
- [ ] `inputs:` in procedures lists key **names only** (e.g., `[employee_id]`), not full definitions
- [ ] `steps:` is followed by ` |` on the same line
- [ ] Steps text is indented further than `steps:`
- [ ] Input values are referenced as `<key>` in steps
- [ ] Output values to capture are referenced as `{key}` in steps
- [ ] All key names are unique across inputs and outputs; each key holds exactly one value
- [ ] Scroll steps are written explicitly before any element that may be off-screen
- [ ] Ambiguous UI elements are identified by their neighboring labeled elements
- [ ] Each `verify:` line names a specific element, text, or screen state
