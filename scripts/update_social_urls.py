"""Update X/LinkedIn social posts with exact WordPress and GitHub Pages URLs."""
import re
from pathlib import Path

REPORTS = Path("/home/user/Documents/Personal/ai_lab/mtdna_foundation_model/reports")

# Exact URLs per blog
BLOGS = {
    1: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/this-genome-has-16569-base-pairs-and-runs-on-different-rules/",
        "gh": "https://vthawfeek.github.io/2026/06/04/this-genome-has-16569-base-pairs-and-runs-on-different-rules/",
    },
    2: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/i-checked-whether-existing-dna-models-could-handle-mitochondrial-dna-heres-the-problem/",
        "gh": "https://vthawfeek.github.io/2026/06/04/i-checked-whether-existing-dna-models-could-handle-mitochondrial-dna-heres-the-p/",
    },
    3: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/three-architecture-decisions-i-made-for-mtdna-fm-and-why-each-one-was-non-obvious/",
        "gh": "https://vthawfeek.github.io/2026/06/04/three-architecture-decisions-i-made-for-mtdna-fm-and-why-each-one-was-non-obviou/",
    },
    4: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/im-training-this-on-a-laptop-cpu-here-is-what-that-forces-me-to-do/",
        "gh": "https://vthawfeek.github.io/2026/06/04/im-training-this-on-a-laptop-cpu-here-is-what-that-forces-me-to-do/",
    },
    5: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/my-haplogroup-classifier-scores-1-83-random-guessing-would-score-3-85/",
        "gh": "https://vthawfeek.github.io/2026/06/04/my-haplogroup-classifier-scores-183-random-guessing-would-score-385/",
    },
    6: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/i-built-a-pathogenicity-predictor-i-dont-have-the-data-to-evaluate-it/",
        "gh": "https://vthawfeek.github.io/2026/06/04/i-built-a-pathogenicity-predictor-i-dont-have-the-data-to-evaluate-it/",
    },
    7: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/what-id-do-differently-if-i-built-mtdna-fm-again/",
        "gh": "https://vthawfeek.github.io/2026/06/04/what-id-do-differently-if-i-built-mtdna-fm-again/",
    },
    8: {
        "wp": "https://rokpayprsizors.wordpress.com/2026/06/04/what-a-first-of-its-kind-foundation-model-looks-like-when-built-in-4-weeks-on-a-laptop/",
        "gh": "https://vthawfeek.github.io/2026/06/04/what-a-first-of-its-kind-foundation-model-looks-like-when-built-in-4-weeks-on-a-/",
    },
}

# Map each social file to its blog number
MAPPING = {
    # Theme 1 - Biology (Blog 1)
    "x-t1-01-circular-topology.txt": 1,
    "x-t1-02-heteroplasmy.txt": 1,
    "x-t1-03-phylogenetic-density.txt": 1,
    "linkedin-t1-01-circular-topology.txt": 1,
    "linkedin-t1-02-heteroplasmy.txt": 1,
    "linkedin-t1-03-phylogenetic-density.txt": 1,
    # Theme 1 - Existing models (Blog 2)
    "x-t1-04-existing-models.txt": 2,
    "linkedin-t1-04-existing-models.txt": 2,
    # Theme 2 - Architecture (Blog 3)
    "x-t2-01-circular-pe.txt": 3,
    "x-t2-02-het-channel.txt": 3,
    "x-t2-03-tokenization.txt": 3,
    "linkedin-t2-01-circular-pe.txt": 3,
    "linkedin-t2-02-het-channel.txt": 3,
    "linkedin-t2-03-tokenization.txt": 3,
    # Theme 2 - CPU training (Blog 4)
    "x-t2-04-two-phase.txt": 4,
    "x-t2-05-cpu-constraints.txt": 4,
    "x-t3-03-dataloader-deadlock.txt": 4,
    "linkedin-t2-04-two-phase.txt": 4,
    "linkedin-t2-05-cpu-constraints.txt": 4,
    "linkedin-t3-03-dataloader-deadlock.txt": 4,
    # Theme 3 - Fine-tuning failed (Blog 5)
    "x-t3-01-below-random.txt": 5,
    "x-t3-02-class-collapse.txt": 5,
    "x-t3-04-two-epochs.txt": 5,
    "linkedin-t3-01-below-random.txt": 5,
    "linkedin-t3-02-class-collapse.txt": 5,
    "linkedin-t3-04-two-epochs.txt": 5,
    # Theme 3 - No pathogenicity eval (Blog 6)
    "x-t3-05-no-data.txt": 6,
    "linkedin-t3-05-no-data.txt": 6,
    # Theme 4 - Learnings (Blog 7)
    "x-t4-01-model-lessons.txt": 7,
    "x-t4-02-training-lessons.txt": 7,
    "linkedin-t4-01-model-lessons.txt": 7,
    "linkedin-t4-02-training-lessons.txt": 7,
    # Theme 5 - Achievements (Blog 8)
    "x-t5-01-zeroshot-knn.txt": 8,
    "x-t5-02-ancient-dna.txt": 8,
    "linkedin-t5-01-zeroshot-knn.txt": 8,
    "linkedin-t5-02-ancient-dna.txt": 8,
}

GENERIC_WP = "https://rokpayprsizors.wordpress.com/"
GENERIC_WP_PATTERN = re.compile(r"https://rokpayprsizors\.wordpress\.com/\s*$", re.MULTILINE)


def url_block(blog_num: int) -> str:
    b = BLOGS[blog_num]
    return f"{b['wp']}\n{b['gh']}"


def update_x_post(text: str, blog_num: int) -> str:
    """Replace generic WP URL or add URL block to X thread."""
    urls = url_block(blog_num)

    # If generic WP URL exists anywhere, replace it (and the line it's on) with both URLs
    if GENERIC_WP in text:
        # Replace the generic URL line with specific WP + GH
        text = text.replace(GENERIC_WP, f"{BLOGS[blog_num]['wp']}\n{BLOGS[blog_num]['gh']}")
        return text

    # No existing URL: add a final tweet before [Attach image] note or before end
    attach_match = re.search(r"\n\[Attach image:.*\]", text)
    insert_before = attach_match.start() if attach_match else len(text)

    link_tweet = f"\n---\nFull writeup:\n{urls}\n"
    return text[:insert_before] + link_tweet + text[insert_before:]


def update_linkedin_post(text: str, blog_num: int) -> str:
    """Replace generic WP URL or add URL block to LinkedIn post."""
    urls = url_block(blog_num)

    if GENERIC_WP in text:
        text = text.replace(GENERIC_WP, f"{BLOGS[blog_num]['wp']}\n{BLOGS[blog_num]['gh']}")
        return text

    # No URL: append at end (before any trailing newline)
    stripped = text.rstrip()
    return stripped + f"\n\nFull writeup:\n{BLOGS[blog_num]['wp']}\n{BLOGS[blog_num]['gh']}\n"


updated = 0
for filename, blog_num in MAPPING.items():
    path = REPORTS / filename
    if not path.exists():
        print(f"MISSING: {filename}")
        continue

    original = path.read_text()
    is_x = filename.startswith("x-")
    updated_text = update_x_post(original, blog_num) if is_x else update_linkedin_post(original, blog_num)

    if updated_text != original:
        path.write_text(updated_text)
        print(f"Updated: {filename} -> blog {blog_num}")
        updated += 1
    else:
        print(f"No change: {filename}")

print(f"\nTotal updated: {updated}/{len(MAPPING)}")
