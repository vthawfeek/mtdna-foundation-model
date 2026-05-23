Generate a blog post, X thread, and LinkedIn post from a day report. Writes draft files and prints a preview. Does NOT publish anything.

## Arguments

`$ARGUMENTS` is a day number (e.g. `1`, `4`, `7`).

## Context

Working directory: current project directory.
Config file: `.publish-config.yml` in the project root (read it if present).
Report file: `reports/day-$ARGUMENTS-*.md` (glob — pick the one that matches).
Output files:
- `reports/day-$ARGUMENTS-blog.md`
- `reports/day-$ARGUMENTS-x.txt`
- `reports/day-$ARGUMENTS-linkedin.txt`

## Steps

1. Read the report at `reports/day-$ARGUMENTS-*.md`.
2. Read `.publish-config.yml` for project name, URL, author, X handle, and default tags. If the file is missing, use sensible defaults from the report content.
3. Generate the blog post, X thread, and LinkedIn post following the style rules below.
4. Write each to its output file.
5. Print a clearly labelled preview of all three to the terminal.
6. Stage and commit the three output files:
   ```
   git add reports/day-$ARGUMENTS-blog.md reports/day-$ARGUMENTS-x.txt reports/day-$ARGUMENTS-linkedin.txt
   git commit -m "day $ARGUMENTS: add content drafts"
   git push origin main
   ```
7. Print exactly this message at the end:
   ```
   Content drafts committed. Review and edit the files if needed:
     reports/day-$ARGUMENTS-blog.md
     reports/day-$ARGUMENTS-x.txt
     reports/day-$ARGUMENTS-linkedin.txt
   When satisfied, run: /blog-send $ARGUMENTS
   ```

## Blog Post Style Rules

**Title:** Name the specific thing built or discovered. No "Day N Update" or "Progress Report".
- Wrong: "Day 8: Model Architecture"
- Right: "Circular Positional Encoding for Mitochondrial DNA: Why Standard BERT Fails"

**Opening paragraph:** Start with the central problem, not a roadmap of the post.
- Wrong: "In this post, I'll walk through how I built..."
- Right: "Position 1 and position 16,569 of the human mitochondrial genome are physically adjacent. Standard BERT positional embeddings treat them as 16,568 positions apart."

**Body:**
- Mix code blocks (use actual function names from the report) with prose explanation
- Include real numbers from the report (loss values, accuracy %, wall-clock times)
- When a design decision was made, name the alternatives that were rejected and explain why
- Be honest about what was harder than expected
- No vague success claims; use specific metrics

**Format:**
- Markdown. First line must be `# {title}` with no leading blank line.
- Length: 1,200-2,000 words
- Short paragraphs, 3-4 sentences max

**Writing rules (apply to ALL three formats):**
- No em-dash (—). Use a comma, colon, or break into two sentences.
- No "delve", "leverage", "robust", "seamlessly", "at the intersection of", "it's worth noting", "dive into"
- No passive voice for findings; "the model achieves X" not "it was found that X"
- No intro sentence that describes what the post will cover ("In this post...", "Today I'll explain...")
- No closing sentence that summarises what was covered ("In conclusion...", "To summarise...")

## X Thread Style Rules

**Tweet 1 (hook):** The most surprising or specific result from the day. Not an announcement.
- Wrong: "Excited to share that I've been building an mtDNA foundation model!"
- Right: "Circular genomes break BERT. Position 1 and 16,569 in mtDNA are adjacent, but standard PE treats them as 16,568 apart. Here's the fix:"

**Tweets 2-4:** One concrete point per tweet: a code snippet, a specific number, or a non-obvious insight.

**Last tweet:** One sentence of context, then `{blog_url}` on its own line.

**Format rules:**
- Separate tweets with `---` on its own line (no blank lines around the separator)
- Each tweet must be 280 characters or fewer
- Max 2 hashtags across the entire thread; only add them if they fit naturally
- No em-dash. No "Excited to share". No "Thrilled to announce".

## LinkedIn Post Style Rules

**Length:** 150-200 words total.

**First sentence:** The most specific finding or result from the day, stated as a fact or observation. Same hook rule as X tweet 1.

**Structure:** 2-3 short paragraphs:
1. What was built and the key technical challenge
2. What was non-obvious or surprising
3. What comes next (one sentence)

**Closing line:** `Full writeup: {blog_url}`

**Rules:**
- No em-dash
- No "I'm excited to share", "Let's connect", "I'm passionate about"
- Use prose for the body, not a bulleted list of everything
- Professional tone; no corporate clichés
