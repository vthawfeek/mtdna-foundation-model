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

3a. After writing the report, generate the day's dashboard data file at:
    `reports/day-$ARGUMENTS-data.js`

    This file registers the day's content on `window.__dayData` so the dashboard can
    load it without embedding all days' content at generation time. Write the file with
    this exact structure:

    ```js
    (function () {
      window.__dayData = window.__dayData || {};
      window.__dayData[$ARGUMENTS] = {
        topic: "<short topic, e.g. 'Tokenizer'>",
        commit: "<7-char commit hash, or empty string if not yet committed>",
        status: "complete",
        built: [
          "<one string per bullet from '## What was built'>",
          ...
        ],
        learned: [
          "<one string per bullet from '## What was learned'>",
          ...
        ],
        decisions: [
          "<one string per bullet from '## Key decisions'>",
          ...
        ],
        eli5: `<div class="eli5-wrap">
          <div class="eli5-art">
            <svg width="120" height="130" viewBox="0 0 120 130">
              <!-- A simple SVG illustration representing the day's core concept -->
            </svg>
            <div class="eli5-caption">One-line caption</div>
          </div>
          <div class="eli5-text">
            <div class="eli5-step"><span class="eli5-num">1</span>Step 1 explanation (2-3 sentences, plain English, no assumed knowledge)</div>
            <div class="eli5-step"><span class="eli5-num">2</span>Step 2 explanation</div>
            <div class="eli5-step"><span class="eli5-num">3</span>Step 3 explanation</div>
            <div class="eli5-step"><span class="eli5-num">4</span>Step 4 explanation</div>
          </div>
        </div>`,
        math: `<div class="math-block">
          <div class="math-heading">Key formula or algorithm name</div>
          <div class="math-eq">the equation or pseudocode, plain text, no LaTeX</div>
          <div class="math-example"><strong>Example:</strong> worked example showing the formula in action</div>
        </div>`
      };
    })();
    ```

    Guidelines for the ELI5 and Math content:
    - ELI5: Explain the day's most important concept as if to a curious 12-year-old. Use
      analogies. Avoid assumed ML/bio knowledge. 4 numbered steps.
    - SVG: Simple geometric illustration (circles, rectangles, arrows, text). No external
      references. Keep it small (120×130 viewBox). Inline styles only.
    - Math: Show the actual equation(s) or algorithm the day implemented. Use plain-text
      notation (not LaTeX). Include one concrete worked example with real numbers.
    - If a day has multiple key formulas, add multiple `<div class="math-block">` sections.
    - Use `&amp;`, `&lt;`, `&gt;` for HTML entities inside regular strings. Inside
      template literals, use `&amp;` only where the literal `&` would cause issues.

4. Run the end-of-day quality checks:
   ```
   uv run ruff check mtdna_fm/ tests/
   uv run pytest tests/ -m "not slow and not integration"
   ```
   Both must pass before committing. Fix any failures first.

5. Mark the day complete in CLAUDE.md: find the line for Day $ARGUMENTS in the "Current status" section and add "COMPLETE (commit <hash>)" to it.

6. Stage and commit:
   - Stage: all new and modified files under mtdna_fm/, tests/, configs/, docs/, notebooks/, reports/, .claude/, .github/, PLAN.md, CLAUDE.md
   - This includes `reports/day-$ARGUMENTS-data.js` (created in step 3a)
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

8. If Day $ARGUMENTS is in the milestone list [1, 4, 7, 10, 13, 14, 17, 20, 21, 22, 25, 26, 28],
   generate content drafts by following all the steps in `~/.claude/commands/blog-draft.md` for Day $ARGUMENTS.

   The blog-draft steps will:
   - Read the report just written and `.publish-config.yml`
   - Generate a blog post, X thread, and LinkedIn post
   - Write them to `reports/day-$ARGUMENTS-blog.md`, `reports/day-$ARGUMENTS-x.txt`, `reports/day-$ARGUMENTS-linkedin.txt`
   - Commit and push the three draft files
   - Print a preview

   Do NOT call `publish.py social` or `publish.py blog` here. Stop after drafting.
   The day is not complete until the content preview has been printed.

## Important

- Follow the plan precisely. The plan has specific file names, function signatures, and architecture decisions — use them.
- If a day is marked [COMPLETE] in the plan, skip the implementation tasks but still create the report if it does not already exist, then commit and push.
- If the plan references a sibling project (`sc_foundation_model`), read the relevant files there before implementing the analogous component here.
- Do not invent tasks that are not in the plan. Do not skip tasks that are in the plan.
