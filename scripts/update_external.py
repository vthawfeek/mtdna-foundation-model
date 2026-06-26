"""
Update external surfaces (HuggingFace model card + WordPress posts) in-place
with the corrected scientific content.

Usage:
    set HF_TOKEN=hf_...
    set WORDPRESS_USERNAME=your_wp_username
    set WORDPRESS_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx
    uv run python scripts/update_external.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
BLOG_FILES = {
    "blog-01": REPO_ROOT / "reports" / "blog-01-mtdna-biology.md",
    "blog-03": REPO_ROOT / "reports" / "blog-03-architecture.md",
    "blog-04": REPO_ROOT / "reports" / "blog-04-cpu-training.md",
    "blog-05": REPO_ROOT / "reports" / "blog-05-finetuning-failed.md",
    "blog-07": REPO_ROOT / "reports" / "blog-07-learnings.md",
    "blog-08": REPO_ROOT / "reports" / "blog-08-achievements.md",
}
WP_SITE = "rokpayprsizors.wordpress.com"


def _get_post_url(blog_file: Path) -> str | None:
    """Extract the <!-- published: URL --> comment from a blog file."""
    text = blog_file.read_text(encoding="utf-8")
    m = re.search(r"<!--\s*published:\s*(https?://\S+)\s*-->", text)
    return m.group(1) if m else None


def _slug_from_url(url: str) -> str:
    """Extract WordPress post slug from URL."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def _get_post_id(slug: str, wp_api_base: str, auth: tuple) -> int | None:
    """Look up post ID by slug via WP REST API."""
    r = requests.get(
        f"{wp_api_base}/posts",
        params={"slug": slug, "status": "publish"},
        auth=auth,
        timeout=30,
    )
    r.raise_for_status()
    posts = r.json()
    return posts[0]["id"] if posts else None


def _markdown_to_html_simple(md: str) -> str:
    """
    Minimal Markdown → HTML for WordPress block editor.
    Handles headings, paragraphs, bold, italic, inline code, fenced code blocks,
    images, and links — enough for these blog posts.
    """
    lines = md.splitlines()
    html_lines = []
    in_code = False
    code_buf: list[str] = []
    code_lang = ""

    for line in lines:
        # Strip published comment
        if line.strip().startswith("<!-- published:"):
            continue

        # Fenced code blocks
        if line.startswith("```"):
            if not in_code:
                in_code = True
                code_lang = line[3:].strip()
                code_buf = []
            else:
                code = "\n".join(code_buf)
                code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_lines.append(
                    f'<!-- wp:code -->\n<pre class="wp-block-code"><code>{code}</code></pre>\n<!-- /wp:code -->'
                )
                in_code = False
                code_buf = []
            continue

        if in_code:
            code_buf.append(line)
            continue

        # Headings
        if line.startswith("### "):
            html_lines.append(f"<!-- wp:heading {{\"level\":3}} -->\n<h3>{_inline(line[4:])}</h3>\n<!-- /wp:heading -->")
        elif line.startswith("## "):
            html_lines.append(f"<!-- wp:heading -->\n<h2>{_inline(line[3:])}</h2>\n<!-- /wp:heading -->")
        elif line.startswith("# "):
            # Skip title — WP uses the post title field
            pass
        elif line.startswith("---"):
            html_lines.append("<!-- wp:separator -->\n<hr class=\"wp-block-separator\"/>\n<!-- /wp:separator -->")
        elif line.startswith("!["):
            # Image: ![alt](src)
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
            if m:
                alt, src = m.group(1), m.group(2)
                html_lines.append(
                    f'<!-- wp:image -->\n<figure class="wp-block-image"><img src="{src}" alt="{alt}"/></figure>\n<!-- /wp:image -->'
                )
        elif line.strip() == "":
            html_lines.append("")
        else:
            html_lines.append(f"<!-- wp:paragraph -->\n<p>{_inline(line)}</p>\n<!-- /wp:paragraph -->")

    return "\n".join(html_lines)


def _inline(text: str) -> str:
    """Convert inline Markdown (bold, italic, code, links) to HTML."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Links
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    return text


# ---------------------------------------------------------------------------
# HuggingFace model card update
# ---------------------------------------------------------------------------

def update_hf_model_card() -> None:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("SKIP HuggingFace: set HF_TOKEN env var first.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    card_path = REPO_ROOT / "models" / "phase1_v1" / "README.md"
    print(f"Uploading corrected model card → vthawfeek/mtdna-foundation-model ...")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id="vthawfeek/mtdna-foundation-model",
        repo_type="model",
        commit_message="fix: correct zero-shot claim (8-class panel, 12.5% random) and dataset size (34,975 used, 117,615 Phase 1)",
    )
    print("✓ HuggingFace model card updated.")


# ---------------------------------------------------------------------------
# WordPress blog post updates
# ---------------------------------------------------------------------------

def update_wordpress_posts() -> None:
    wp_user = os.environ.get("WORDPRESS_USERNAME")
    wp_pass = os.environ.get("WORDPRESS_APP_PASSWORD")

    if not wp_user or not wp_pass:
        print("SKIP WordPress: set WORDPRESS_USERNAME and WORDPRESS_APP_PASSWORD env vars first.")
        return

    auth = (wp_user, wp_pass)
    wp_api_base = f"https://public-api.wordpress.com/wp/v2/sites/{WP_SITE}"

    for key, blog_file in BLOG_FILES.items():
        url = _get_post_url(blog_file)
        if not url:
            print(f"  SKIP {key}: no published URL found in file.")
            continue

        slug = _slug_from_url(url)
        print(f"  {key}: looking up post '{slug}' ...")
        post_id = _get_post_id(slug, wp_api_base, auth)
        if not post_id:
            print(f"  SKIP {key}: post not found via API (slug={slug}).")
            continue

        raw = blog_file.read_text(encoding="utf-8")
        # Strip the published comment before converting
        content_without_comment = re.sub(r"\n?<!--\s*published:.*?-->\s*$", "", raw, flags=re.DOTALL)
        html_content = _markdown_to_html_simple(content_without_comment)

        r = requests.post(
            f"{wp_api_base}/posts/{post_id}",
            json={"content": html_content},
            auth=auth,
            timeout=60,
        )
        r.raise_for_status()
        print(f"  ✓ {key}: updated → {url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Updating external surfaces ===\n")
    update_hf_model_card()
    print()
    update_wordpress_posts()
    print("\nDone.")
