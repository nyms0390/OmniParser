---
name: pack
description: Zip changed source files and email them to a recipient. Use when someone needs the actual files (e.g. no git access). Trigger phrases include "pack changes", "zip changed files", "export changes", "pack for transfer", "pack the latest commit". NOT for preserving work context — use /handoff for that.
version: 1.1.0
disable-model-invocation: true
allowed-tools: [Bash]
---

Zip changed source files and deliver them via email.

**Use this when:** a recipient needs the actual source files and doesn't have git access.
**Not this skill:** if you need to document progress for the next agent/session — use `/handoff` for that.

## Exclusions

Always exclude:
- Files matched by `.gitignore` (use `git ls-files --ignored --exclude-standard` to check)
- Claude config files: anything under `.claude/` (settings, memory, skills, etc.)

## Current State
- Branch: !`git branch --show-current`
- Changed files (committed on branch): !`git diff --name-only master...HEAD`
- Uncommitted changes: !`git diff --name-only HEAD`
- Untracked files: !`git ls-files --others --exclude-standard`

## Scope

- **"Pack the latest commit"** → use only the files from `git diff-tree --no-commit-id -r --name-only HEAD`
- **"Pack changes"** / **"Pack branch"** → combine all three lists above into a deduplicated set; skip deleted files

After collecting the file list, filter out:
1. Any path starting with `.claude/`
2. Any path that is gitignore'd: `git check-ignore -q <file> && echo ignored`

## Steps

1. **Collect files** per the scope above, then apply the exclusions.

2. **Create the archive** from the repo root. Use a descriptive slug and timestamp in the filename:
   ```bash
   ZIPFILE=~/Desktop/omniparser-<slug>-<YYYYMMDD-HHMM>.zip
   # Filter: remove .claude/ paths and gitignore'd files
   FILES=$(echo "<file list>" | tr ' ' '\n' | grep -v '^\.claude/' | while read f; do
     git check-ignore -q "$f" || echo "$f"
   done | tr '\n' ' ')
   zip "$ZIPFILE" $FILES   # no -r needed for explicit file lists
   echo "Created: $ZIPFILE"
   ```

3. **Send via email** using the helper at `~/.claude/scripts/send_email.py`:
   ```bash
   python - <<'EOF'
   import sys, os
   sys.path.insert(0, os.path.expanduser('~/.claude/scripts'))
   from send_email import send_email_with_attachment
   send_email_with_attachment(
       to_email='nyms.chang@ctbcbank.com',
       from_email='nyms0390@gmail.com',
       subject='OmniParser changes — <zip filename>',
       body='Changed files from <scope description>.',
       attachment_path='<full zip path>',
       smtp_server='smtp.gmail.com',
       smtp_port=587,
       username='nyms0390@gmail.com',
       password=os.environ['GMAIL_APP_PASSWORD'],
   )
   EOF
   ```
   If `GMAIL_APP_PASSWORD` is not set in the environment, stop and tell the user.

4. **Report**:
   - Files included and total count
   - Zip path
   - Confirmation that the email was sent (or the error if it failed)
