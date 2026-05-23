Generate a daily HTML dashboard for the current project and open it in the browser.

## Context

Working directory: current project directory.
Config: `.publish-config.yml` (read if present for project name).
Output: `dashboard.html` in the project root.

## Steps

### 1. Extract project data

Read `CLAUDE.md` and find all lines in the "Current status" section. Parse them to find:
- `LAST_DAY`: the highest day number that has `COMPLETE` in its line
- `LAST_DAY_TOPIC`: the topic name from that line
- `LAST_DAY_COMMIT`: the commit hash from that line (e.g. `commit 12d614c`)
- If no days are complete, set `LAST_DAY = 0`

Set `TODAY_DAY = LAST_DAY + 1`.

Read `PLAN.md` and locate `### Day TODAY_DAY:` and `### Day (TODAY_DAY+1):`. Extract:
- `TODAY_TOPIC`: the heading text after "Day N:"
- `TODAY_TASKS`: all bullet points under that heading (stop at next `###` or `---`)
- `NEXT_TOPIC`: heading text of the following day
- `NEXT_TASKS`: first 3 bullet points of the following day

Read `reports/day-LAST_DAY-*.md` (glob — pick the matching file):
- Extract everything under `## What was learned` up to the next `##` section
- Call this `YESTERDAY_LEARNINGS` (list of bullet points, max 6)

Check if `reports/day-TODAY_DAY-*.md` exists:
- If yes: extract `## What was learned` bullets → `TODAY_LEARNINGS`
- If no: `TODAY_LEARNINGS = []`, set `TODAY_NOT_STARTED = true`

Check for pending social media by scanning `reports/` for files matching `day-*-blog.md`:
- For each, check if it contains a line starting with `<!-- published:` → if not, it is pending
- Build a list: `[{day: N, command: "/blog-send N"}, ...]`

Calculate `PROGRESS_PCT = round(LAST_DAY / 28 * 100)`.

Read `PLAN.md` to find the week label for TODAY_DAY:
- Days 1-7: "Week 1: Infrastructure and Data"
- Days 8-14: "Week 2: Model Architecture"
- Days 15-21: "Week 3: Fine-tuning and Evaluation"
- Days 22-28: "Week 4: Production Polish"

### 2. Compute recommended posting dates

From today's date:
- `X_POST_DATE`: next weekday (Mon-Fri), formatted as "Mon 25 May"
- `LI_POST_DATE`: next Tuesday, Wednesday, or Thursday — whichever comes first, formatted same way

### 3. Write dashboard.html

Write a complete, self-contained HTML file to `dashboard.html`. Use the exact structure and styling below, substituting the extracted values.

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>mtDNA Foundation Model — Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; color: #1a1a2e; padding: 24px; }
  .header { background: #1a1a2e; color: #fff; border-radius: 12px;
             padding: 20px 28px; margin-bottom: 20px; display: flex;
             justify-content: space-between; align-items: center; }
  .header h1 { font-size: 1.3rem; font-weight: 600; }
  .header .date { opacity: 0.7; font-size: 0.9rem; }
  .progress-bar { background: #2d2d4e; border-radius: 6px; height: 8px;
                  margin-top: 12px; overflow: hidden; }
  .progress-fill { background: #4ade80; height: 100%; border-radius: 6px;
                   width: {PROGRESS_PCT}%; transition: width 0.5s; }
  .progress-label { margin-top: 6px; font-size: 0.78rem; opacity: 0.7; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-full { grid-column: 1 / -1; }
  .card { background: #fff; border-radius: 12px; padding: 20px 24px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .card-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
                text-transform: uppercase; margin-bottom: 8px; color: #888; }
  .card-title { font-size: 1rem; font-weight: 600; margin-bottom: 12px; }
  .task-list { list-style: none; }
  .task-list li { padding: 5px 0; font-size: 0.88rem; display: flex;
                  align-items: flex-start; gap: 8px; color: #333; }
  .task-list li::before { content: "□"; color: #93c5fd; font-size: 1rem;
                           flex-shrink: 0; margin-top: -1px; }
  .learning-list { list-style: none; }
  .learning-list li { padding: 5px 0; font-size: 0.88rem; color: #333;
                      padding-left: 16px; position: relative; }
  .learning-list li::before { content: "•"; position: absolute; left: 0;
                               color: #a78bfa; }
  .pending-item { background: #fffbeb; border: 1px solid #fcd34d;
                  border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
                  font-size: 0.88rem; }
  .pending-cmd { font-family: monospace; background: #fef3c7;
                 padding: 2px 6px; border-radius: 4px; }
  .next-action { background: #1a1a2e; color: #fff; border-radius: 12px;
                 padding: 16px 24px; margin-top: 16px; display: flex;
                 align-items: center; justify-content: space-between; }
  .next-action .label { font-size: 0.75rem; text-transform: uppercase;
                        letter-spacing: 0.08em; opacity: 0.6; }
  .next-action .cmd { font-family: monospace; font-size: 1.1rem;
                      color: #4ade80; font-weight: 700; }
  .done-badge { background: #dcfce7; color: #16a34a; font-size: 0.75rem;
                font-weight: 600; padding: 2px 8px; border-radius: 12px; }
  .week-label { font-size: 0.78rem; color: #888; margin-top: 4px; }
  .no-pending { color: #888; font-size: 0.88rem; font-style: italic; }
  .empty-learnings { color: #aaa; font-size: 0.88rem; font-style: italic; }
  @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>{PROJECT_NAME}</h1>
    <div class="date">{TODAY_DATE} &nbsp;·&nbsp; {WEEK_LABEL}</div>
    <div class="progress-bar"><div class="progress-fill"></div></div>
    <div class="progress-label">Day {LAST_DAY} of 28 complete &nbsp;·&nbsp; {PROGRESS_PCT}%</div>
  </div>
</div>

<div class="grid">

  <div class="card">
    <div class="card-label">Last Completed</div>
    {IF LAST_DAY > 0}
    <div class="card-title">Day {LAST_DAY}: {LAST_DAY_TOPIC} <span class="done-badge">✓</span></div>
    <div style="font-size:0.8rem;color:#888">{LAST_DAY_COMMIT}</div>
    {ELSE}
    <div class="card-title" style="color:#aaa">Nothing completed yet</div>
    {END IF}
  </div>

  <div class="card">
    <div class="card-label">What's Next — Day {TODAY_DAY + 1}</div>
    <div class="card-title">{NEXT_TOPIC}</div>
    <ul class="task-list">
      {FOR EACH task IN NEXT_TASKS (max 3)}
      <li>{task}</li>
      {END FOR}
    </ul>
  </div>

  <div class="card grid-full">
    <div class="card-label">Today — Day {TODAY_DAY}</div>
    <div class="card-title">{TODAY_TOPIC}</div>
    <ul class="task-list">
      {FOR EACH task IN TODAY_TASKS}
      <li>{task}</li>
      {END FOR}
    </ul>
  </div>

  <div class="card grid-full">
    <div class="card-label">Pending Social Media</div>
    {IF PENDING_SOCIAL is empty}
    <div class="no-pending">No drafts pending.</div>
    {ELSE}
    {FOR EACH item IN PENDING_SOCIAL}
    <div class="pending-item">
      Day {item.day} draft ready &nbsp;→&nbsp;
      run <span class="pending-cmd">{item.command}</span> to publish and schedule
    </div>
    {END FOR}
    {END IF}
  </div>

  <div class="card">
    <div class="card-label">Yesterday's Learnings — Day {LAST_DAY}</div>
    {IF LAST_DAY == 0}
    <div class="empty-learnings">No completed days yet.</div>
    {ELSE IF YESTERDAY_LEARNINGS is empty}
    <div class="empty-learnings">No report found for Day {LAST_DAY}.</div>
    {ELSE}
    <ul class="learning-list">
      {FOR EACH learning IN YESTERDAY_LEARNINGS}
      <li>{learning}</li>
      {END FOR}
    </ul>
    {END IF}
  </div>

  <div class="card">
    <div class="card-label">Today's Learnings — Day {TODAY_DAY}</div>
    {IF TODAY_NOT_STARTED}
    <div class="empty-learnings">Run <code>/day {TODAY_DAY}</code> to start.</div>
    {ELSE IF TODAY_LEARNINGS is empty}
    <div class="empty-learnings">In progress — no learnings recorded yet.</div>
    {ELSE}
    <ul class="learning-list">
      {FOR EACH learning IN TODAY_LEARNINGS}
      <li>{learning}</li>
      {END FOR}
    </ul>
    {END IF}
  </div>

</div>

<div class="next-action">
  <span class="label">Next Action</span>
  <span class="cmd">/day {TODAY_DAY}</span>
</div>

</body>
</html>
```

Substitute all `{PLACEHOLDER}` values with the actual extracted data. The template syntax above (`{IF}`, `{FOR EACH}`) is pseudocode — render the actual HTML with the real values, not template tags.

### 4. Open in browser

```bash
xdg-open dashboard.html 2>/dev/null || open dashboard.html 2>/dev/null || echo "Dashboard written. Open dashboard.html in your browser."
```

Print: `Dashboard generated: dashboard.html`
