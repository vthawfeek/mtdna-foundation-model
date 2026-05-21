Execute Day $ARGUMENTS of the mtDNA Foundation Model project.

## Context

Working directory: /home/user/Documents/Personal/ai_lab/mtdna_foundation_model
Plan file: PLAN.md (in the repo root — this is the primary copy tracked in git)
GitHub repo: https://github.com/vthawfeek/mtdna-foundation-model

## Instructions

1. Read PLAN.md and find the section titled "### Day $ARGUMENTS:". Execute every task listed there, in order. Do not skip tasks. If a task produces an error, fix the error before moving on.

2. For each source file created or modified, follow these rules:
   - All Python code must pass `uv run ruff check mtdna_fm/ tests/`
   - All tests must pass `uv run pytest tests/ -m "not slow and not integration"`
   - Fix any failures before proceeding to the next task

3. After all tasks are complete, write a daily report at:
   `reports/day-$ARGUMENTS-<short-topic>.md`

   Use this exact structure:
   ```
   # Day $ARGUMENTS: <Topic>

   ## What was built
   - <bullet per file or component created/modified>

   ## What was learned
   - <bullet per concept this day covered and why it matters>

   ## Key decisions
   - <decision>: <why>

   ## Verification
   <commands run to confirm correctness and their output>

   ## Next up
   Day <$ARGUMENTS+1>: <one sentence on what comes next>
   ```

4. Run the end-of-day quality checks:
   ```
   uv run ruff check mtdna_fm/ tests/
   uv run pytest tests/ -m "not slow and not integration"
   ```
   Both must pass before committing. Fix any failures first.

5. Mark the day complete in CLAUDE.md: find the line for Day $ARGUMENTS in the "Current status" section and add "COMPLETE (commit <hash>)" to it.

6. Stage and commit:
   - Stage: all new and modified files under mtdna_fm/, tests/, configs/, docs/, notebooks/, reports/, .claude/, .github/, PLAN.md, CLAUDE.md
   - Do NOT stage: data/, models/, mlruns/, .venv/, *.pyc, __pycache__/
   - Commit message format: `day $ARGUMENTS: <short description of what was built>`
   - Add co-author: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

7. Push to GitHub:
   ```
   git push origin main
   ```
   If the push fails because there is no remote, remind the user to run:
   ```
   git remote add origin https://github.com/vthawfeek/mtdna-foundation-model.git
   git push -u origin main
   ```

## Important

- Follow the plan precisely. The plan has specific file names, function signatures, and architecture decisions — use them.
- If a day is marked [COMPLETE] in the plan, skip the implementation tasks but still create the report if it does not already exist, then commit and push.
- If the plan references a sibling project (`sc_foundation_model`), read the relevant files there before implementing the analogous component here.
- Do not invent tasks that are not in the plan. Do not skip tasks that are in the plan.
