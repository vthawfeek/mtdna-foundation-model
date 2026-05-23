Publish the blog draft to WordPress.com (and GitHub Pages if configured), then print the X thread and LinkedIn post for manual posting. Requires draft files created by /blog-draft.

## Arguments

`$ARGUMENTS` is a day number (e.g. `1`, `4`, `7`).

## Prerequisites

Draft files must exist (created by `/blog-draft $ARGUMENTS`):
- `reports/day-$ARGUMENTS-blog.md`
- `reports/day-$ARGUMENTS-x.txt`
- `reports/day-$ARGUMENTS-linkedin.txt`

Environment variables in `~/.env`:
- `WORDPRESS_USERNAME` and `WORDPRESS_APP_PASSWORD`

## Steps

1. Verify the three draft files exist. If any is missing, print:
   ```
   Missing: reports/day-$ARGUMENTS-blog.md (or x.txt / linkedin.txt)
   Run /blog-draft $ARGUMENTS first.
   ```
   Then stop.

2. Check if `reports/day-$ARGUMENTS-blog.md` already contains a line starting with `<!-- published:`. If yes, extract the URL from that line and skip to step 5 (already published).

3. Extract the blog post title from the first line of `reports/day-$ARGUMENTS-blog.md` (strip the leading `# `).

4. Extract tags from `.publish-config.yml` if present, otherwise use `mtDNA,foundation model,bioinformatics`.

5. Publish the blog post:
   ```
   uv run python ~/.claude/tools/publish.py blog \
     --title "<extracted title>" \
     --content-file reports/day-$ARGUMENTS-blog.md \
     --tags "<tags>" \
     --site "rokpayprsizors.wordpress.com"
   ```
   Capture the printed URL as BLOG_URL. If the command fails or prints no URL, stop and show the error.
   
   After success, append `<!-- published: BLOG_URL -->` as the last line of `reports/day-$ARGUMENTS-blog.md` so re-runs are skipped.

6. Replace `{blog_url}` in the X and LinkedIn draft files with BLOG_URL (read into memory — do not overwrite the files).

7. Compute posting dates from today:
   - X_DATE: next weekday (Mon-Fri), formatted as "Weekday DD Month YYYY"
   - LI_DATE: next Tuesday, Wednesday, or Thursday — whichever comes first, same format

8. Print this block exactly (substituting real values):

```
✓ Published: <BLOG_URL>

━━ X THREAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Post on X at: <X_DATE>, 9am your time
→ https://x.com/vthawfeek

[tweet 1 text]

↩ reply:
[tweet 2 text]

↩ reply:
[tweet N text]

━━ LINKEDIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Post on LinkedIn at: <LI_DATE>, 9am your time
→ https://www.linkedin.com/in/tvarusai/

[full linkedin post text]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The X thread is split on `---` separators — print each tweet separated by `↩ reply:`.

## Notes

- If `uv` is not available, fall back to `python3 ~/.claude/tools/publish.py`.
- GitHub Pages publishing is handled inside `publish.py` if `GITHUB_PAGES_REPO` is set in `~/.env` — no extra step needed here.
