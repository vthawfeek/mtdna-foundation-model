"""Prepend a varied series-context line to every X thread and LinkedIn post."""
from pathlib import Path

REPORTS = Path("/home/user/Documents/Personal/ai_lab/mtdna_foundation_model/reports")

T0_WP = "https://rokpayprsizors.wordpress.com/2026/06/04/im-going-to-build-a-foundation-model-on-my-laptop/"
T0_GH = "https://vthawfeek.github.io/2026/06/04/im-going-to-build-a-foundation-model-on-my-laptop/"
T0_LINK = f"{T0_WP}\n{T0_GH}"

# (filename, x_context_tweet, linkedin_context_line)
# x_context_tweet: one tweet, max ~230 chars (URL counted separately at ~46 chars)
# linkedin_context_line: one or two sentences ending before a blank line

CONTEXTS = {
    # Theme 1 -- Biology
    "t1-01-circular-topology": (
        f"I'm building a foundation model for mitochondrial DNA on my laptop. No GPU. Here's one reason existing models can't handle it:\n{T0_LINK}",
        f"I'm building a foundation model for mitochondrial DNA on my personal laptop, with no GPU budget. This post is about one of the first problems I ran into.\n{T0_WP}",
    ),
    "t1-02-heteroplasmy": (
        f"Building a foundation model for mtDNA on a laptop CPU. This is one reason I chose this use case over others.\n{T0_LINK}",
        f"I set out to build a foundation model on my laptop with no GPU budget. This is one of the reasons mitochondrial DNA turned out to be the right use case.\n{T0_WP}",
    ),
    "t1-03-phylogenetic-density": (
        f"I set out to build a foundation model on my laptop with no GPU. mtDNA has an unusual property that makes it a strong pre-training target.\n{T0_LINK}",
        f"Building a foundation model on a personal laptop with no cloud compute. Here's why mitochondrial DNA turned out to have unusually rich pre-training signal.\n{T0_WP}",
    ),
    "t1-04-existing-models": (
        f"Building an mtDNA foundation model on my laptop. Before writing any model code, I checked whether existing models could handle this. They can't.\n{T0_LINK}",
        f"I'm building an mtDNA foundation model on my laptop. Before writing any model code, I checked whether existing models could handle this. They can't.\n{T0_WP}",
    ),
    # Theme 2 -- Architecture
    "t2-01-circular-pe": (
        f"I'm building a foundation model for mitochondrial DNA on a laptop CPU. This is one architecture decision that sets it apart from existing genomic models.\n{T0_LINK}",
        f"I'm building a foundation model for mitochondrial DNA on my laptop CPU. This post is about one architecture decision that sets it apart from existing genomic models.\n{T0_WP}",
    ),
    "t2-02-het-channel": (
        f"Building a foundation model for mtDNA on a personal laptop. This design choice came from a biological constraint, not just an engineering preference.\n{T0_LINK}",
        f"Building a foundation model for mtDNA on a personal laptop. This design choice came from a biological constraint, not just an engineering preference.\n{T0_WP}",
    ),
    "t2-03-tokenization": (
        f"I'm building a foundation model for mtDNA on my laptop. Here's why the standard tokenization approach doesn't fit a circular genome.\n{T0_LINK}",
        f"I'm building a foundation model on a laptop with no GPU. This post explains one tokenization decision and why the standard approach doesn't work for a circular genome.\n{T0_WP}",
    ),
    "t2-04-two-phase": (
        f"Training a foundation model on a laptop CPU forces careful choices about training strategy. Here's the one I'm using.\n{T0_LINK}",
        f"Training a foundation model on a laptop CPU requires being deliberate about what you train on and in what order. This is the strategy I'm using.\n{T0_WP}",
    ),
    "t2-05-cpu-constraints": (
        f"I set out to build a foundation model on my laptop with no GPU budget. Here are the actual numbers on what that costs in time and memory.\n{T0_LINK}",
        f"I'm training a foundation model on a personal laptop with no cloud compute. Here are the actual numbers on what that costs in time and memory.\n{T0_WP}",
    ),
    # Theme 3 -- What didn't work
    "t3-01-below-random": (
        f"I'm building a foundation model for mtDNA on my laptop. The fine-tuning results are honest, not flattering.\n{T0_LINK}",
        f"I'm building a foundation model for mitochondrial DNA on my laptop. The fine-tuning results are below random. This post explains what that means and why it happened.\n{T0_WP}",
    ),
    "t3-02-class-collapse": (
        f"Building an mtDNA foundation model on a laptop CPU. This is what class collapse looks like when you can't run enough epochs.\n{T0_LINK}",
        f"Building an mtDNA foundation model on a laptop CPU. This post is about class collapse and what it looks like when compute is the binding constraint.\n{T0_WP}",
    ),
    "t3-03-dataloader-deadlock": (
        f"I'm training a foundation model on a laptop. One DataLoader config mistake cost me 13.5 hours of wall clock time.\n{T0_LINK}",
        f"I'm training a foundation model on a laptop. One mistake in the DataLoader config cost me 13.5 hours of wall clock time. Here's what happened.\n{T0_WP}",
    ),
    "t3-04-two-epochs": (
        f"I'm building a foundation model for mtDNA on my laptop. Here's the math of why 2 fine-tuning epochs isn't enough to converge on CPU.\n{T0_LINK}",
        f"Building a foundation model on a laptop with no GPU. Here's the arithmetic of why 2 fine-tuning epochs isn't enough to converge, and what fixing it actually requires.\n{T0_WP}",
    ),
    "t3-05-no-data": (
        f"Building a foundation model for mtDNA on a laptop. I built a pathogenicity predictor before I had the dataset to evaluate it.\n{T0_LINK}",
        f"I'm building a foundation model for mtDNA on my laptop. I built a pathogenicity predictor before preparing the evaluation dataset. This is what that gap looks like.\n{T0_WP}",
    ),
    # Theme 4 -- Learnings
    "t4-01-model-lessons": (
        f"I built a foundation model for mtDNA on my laptop, no GPU. Here's what I'd change about the architecture.\n{T0_LINK}",
        f"I built a foundation model for mitochondrial DNA on a personal laptop with no GPU budget. Here are the architecture decisions I'd change.\n{T0_WP}",
    ),
    "t4-02-training-lessons": (
        f"I built a foundation model for mtDNA on a laptop CPU over 25 days. These are the training, data, and eval lessons from that.\n{T0_LINK}",
        f"I built a foundation model on a laptop CPU over 25 days. These are the training, data, and evaluation lessons from that sprint.\n{T0_WP}",
    ),
    # Theme 5 -- Achievements
    "t5-01-zeroshot-knn": (
        f"I built a foundation model for mitochondrial DNA on my laptop. The pre-training produced something I didn't fully expect.\n{T0_LINK}",
        f"I built a foundation model for mitochondrial DNA on my personal laptop. The pre-training produced representations I didn't fully expect. Here's the result.\n{T0_WP}",
    ),
    "t5-02-ancient-dna": (
        f"I built a foundation model for mtDNA on a laptop, training on modern genomes only. Then I tested it on something 50,000 years old.\n{T0_LINK}",
        f"I built a foundation model for mtDNA on a laptop CPU, training entirely on modern genomes. Then I tested it on something 50,000 years old.\n{T0_WP}",
    ),
}


def prepend_x(filename_stem: str, text: str) -> str:
    x_tweet, _ = CONTEXTS[filename_stem]
    return x_tweet + "\n---\n" + text


def prepend_linkedin(filename_stem: str, text: str) -> str:
    _, li_line = CONTEXTS[filename_stem]
    return li_line + "\n\n" + text


updated = 0
for stem, (x_ctx, li_ctx) in CONTEXTS.items():
    x_file = REPORTS / f"x-{stem}.txt"
    li_file = REPORTS / f"linkedin-{stem}.txt"

    for path, fn in [(x_file, prepend_x), (li_file, prepend_linkedin)]:
        if not path.exists():
            print(f"MISSING: {path.name}")
            continue
        original = path.read_text()
        updated_text = fn(stem, original)
        if updated_text != original:
            path.write_text(updated_text)
            print(f"Updated: {path.name}")
            updated += 1
        else:
            print(f"No change: {path.name}")

print(f"\nTotal updated: {updated}/36")
