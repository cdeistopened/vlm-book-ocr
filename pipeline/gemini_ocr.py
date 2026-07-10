#!/usr/bin/env python3
"""VLM-based OCR for scanned books: PDF pages -> images -> Gemini -> markdown.

Extracts each PDF page as a JPEG and sends it to Gemini as an image
(not as a PDF — image input avoids some, but not all, RECITATION blocks
on published books; see README "Known gotchas").

Output: one chunk file per batch under <output-dir>/chunks/, each page
wrapped in `=== PAGE NNN ===` markers, ready for stitch.py.
Resumable — re-running skips chunks that already exist on disk.

Requires:
    pip install google-genai pymupdf
    export GEMINI_API_KEY=...   (Google AI Studio key; never hardcode)

Usage:
    python pipeline/gemini_ocr.py book.pdf --model gemini-3-flash-preview --dpi 300 --batch 1
"""

import argparse
import base64
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not installed — run: pip install pymupdf")

try:
    from google import genai
except ImportError:
    sys.exit("google-genai not installed — run: pip install google-genai")


DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_BATCH_SIZE = 1  # one page per call — multi-page batches fail silently
DEFAULT_DPI = 300
MAX_OUTPUT_TOKENS = 65535  # high: thinking tokens count against this budget
THINKING_BUDGET = 512      # low: keep the model from eating the output budget
MAX_RETRIES = 3
RETRY_BACKOFF = 2
RATE_LIMIT_DELAY = 1.0

FALLBACK_RULES = """\
1. VERBATIM: reproduce the text exactly as printed — do not modernize spelling,
   punctuation, or correct apparent errors.
2. DEHYPHENATE within the page: a word split across a line break joins into one
   word. Keep genuine compound hyphens.
3. REFLOW: one line per paragraph. Preserve paragraph boundaries.
4. OMIT the running head and the page folio (number).
5. HEADINGS: ONLY if a heading is printed in the body of THIS page. NEVER infer
   or invent a heading from context — a mid-chapter page gets NO heading.
6. PAGE-BOUNDARY HYPHEN: if the page's last printed word is hyphenated, keep the
   trailing hyphen as-is.
7. FOOTNOTES: transcribe at the end of the page under a line `FOOTNOTES:`,
   each as `^N text`. Keep in-text markers as `[^N]`.
8. Ignore ink marks, stamps, marginalia.
9. Uncertain reading -> [?word].
"""


def load_rules() -> str:
    """Load TRANSCRIBE-RULES.md sitting next to this script, else fallback."""
    rules_path = Path(__file__).parent / "TRANSCRIBE-RULES.md"
    if rules_path.exists():
        return rules_path.read_text()
    return FALLBACK_RULES


def build_prompt(n_pages: int) -> str:
    rules = load_rules()
    plural = "pages" if n_pages > 1 else "page"
    return (
        f"You are transcribing {n_pages} scanned book {plural} for the owner "
        f"of this book, who is producing an accessible text edition.\n\n"
        f"Transcribe each page as markdown following these rules exactly:\n\n"
        f"{rules}\n"
        f"Output ONLY the transcription — no commentary, no code fences. "
        f"If given multiple pages, separate them with a line containing only `---PAGE-BREAK---`."
    )


def get_client() -> "genai.Client":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set. Get a key at https://aistudio.google.com and: export GEMINI_API_KEY=...")
    return genai.Client(api_key=api_key)


def extract_page_as_jpeg(doc, page_num: int, dpi: int) -> bytes:
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    return pix.tobytes("jpeg")


def finish_reason(response) -> str:
    try:
        return str(response.candidates[0].finish_reason)
    except Exception:
        return "UNKNOWN"


def ocr_batch(client, images: list, model: str, retries: int = MAX_RETRIES) -> str:
    """Send a batch of JPEG images to Gemini. Returns markdown text."""
    parts = [
        {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(b).decode()}}
        for b in images
    ]
    parts.append({"text": build_prompt(len(images))})

    config = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1,
        "thinking_config": {"thinking_budget": THINKING_BUDGET},
    }

    last_error = None
    for attempt in range(retries):
        try:
            if attempt > 0:
                backoff = RETRY_BACKOFF ** attempt
                print(f"    [retry {attempt + 1}/{retries} after {backoff}s]")
                time.sleep(backoff)

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": parts}],
                    config=config,
                )
            except Exception as e:
                # Some models reject thinking_config — retry once without it.
                if "thinking" in str(e).lower():
                    config.pop("thinking_config", None)
                    response = client.models.generate_content(
                        model=model,
                        contents=[{"role": "user", "parts": parts}],
                        config=config,
                    )
                else:
                    raise

            reason = finish_reason(response)
            if "RECITATION" in reason:
                # Non-retryable: Gemini refuses books it recognizes from its
                # corpus, regardless of public-domain status. See README.
                raise RuntimeError(
                    "Gemini blocked this page with finishReason=RECITATION "
                    "(the book likely exists in Google's corpus). "
                    "Use the Claude Code path instead — see claude-code/README.md."
                )
            if response.text is None:
                raise RuntimeError(
                    f"Empty response (finishReason={reason}). If MAX_TOKENS, "
                    f"thinking may have eaten the output budget."
                )
            return response.text.strip()

        except RuntimeError as e:
            if "RECITATION" in str(e):
                raise
            last_error = e
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if "rate" in err or "quota" in err or "429" in err:
                wait = RETRY_BACKOFF ** (attempt + 2)
                print(f"    [rate limited, waiting {wait}s]")
                time.sleep(wait)
                continue
            if "500" in err or "503" in err or "timeout" in err:
                continue
            raise

    raise RuntimeError(f"OCR failed after {retries} attempts: {last_error}")


def split_pages(text: str, expected: int) -> list:
    """Split a multi-page response on the page-break sentinel."""
    if expected == 1:
        return [text]
    pages = [p.strip() for p in text.split("---PAGE-BREAK---")]
    return pages


def process_pdf(pdf_path: str, output_dir: str, model: str, dpi: int, batch_size: int,
                first_page: int = 1, last_page: int = 0) -> dict:
    doc = fitz.open(pdf_path)
    total_pdf_pages = len(doc)
    stem = Path(pdf_path).stem

    start = first_page - 1
    end = last_page if last_page else total_pdf_pages
    end = min(end, total_pdf_pages)
    page_nums = list(range(start, end))
    total_pages = len(page_nums)
    if total_pages == 0:
        sys.exit(f"No pages in range {first_page}-{end} (PDF has {total_pdf_pages}).")

    out_dir = Path(output_dir) if output_dir else Path(pdf_path).parent / f"{stem}-ocr"
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    num_batches = math.ceil(total_pages / batch_size)
    client = get_client()

    total_chars = 0
    failed = 0
    t0 = time.time()

    print(f"\n  VLM Book OCR")
    print(f"  ============")
    print(f"  File:    {Path(pdf_path).name}")
    print(f"  Pages:   {total_pages} (of {total_pdf_pages})")
    print(f"  Batches: {num_batches} ({batch_size} page{'s' if batch_size > 1 else ''} each)")
    print(f"  DPI:     {dpi}")
    print(f"  Model:   {model}")
    print(f"  Chunks:  {chunks_dir}\n")

    for b in range(num_batches):
        batch_pages = page_nums[b * batch_size:(b + 1) * batch_size]
        label = f"pages {batch_pages[0] + 1}-{batch_pages[-1] + 1}" if len(batch_pages) > 1 \
            else f"page {batch_pages[0] + 1}"
        chunk_file = chunks_dir / f"{stem}_p{batch_pages[0] + 1:04d}.md"

        # Resume: skip chunks already on disk
        if chunk_file.exists() and chunk_file.stat().st_size > 50:
            print(f"  [{b + 1}/{num_batches}] {label} — cached")
            total_chars += chunk_file.stat().st_size
            continue

        images = [extract_page_as_jpeg(doc, p, dpi) for p in batch_pages]

        try:
            t1 = time.time()
            result = ocr_batch(client, images, model=model)
            page_texts = split_pages(result, len(batch_pages))

            blocks = []
            for pg, txt in zip(batch_pages, page_texts):
                blocks.append(f"=== PAGE {pg + 1:03d} ===\n{txt.strip()}")
            content = "\n\n".join(blocks) + "\n"

            chunk_file.write_text(content)
            total_chars += len(content)
            print(f"  [{b + 1}/{num_batches}] {label} — {len(content):,} chars in {time.time() - t1:.1f}s")

        except Exception as e:
            failed += 1
            print(f"  [{b + 1}/{num_batches}] {label} — ERROR: {e}")
            if "RECITATION" in str(e):
                print("\n  RECITATION block is not retryable. Stopping.")
                print("  Switch to the Claude Code path: see claude-code/README.md")
                break

        progress = {
            "pdf": Path(pdf_path).name,
            "batches_total": num_batches,
            "batches_done": b + 1,
            "failed": failed,
            "elapsed_s": round(time.time() - t0, 1),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        (chunks_dir / "progress.json").write_text(json.dumps(progress, indent=2))

        if b < num_batches - 1:
            time.sleep(RATE_LIMIT_DELAY)

    doc.close()

    elapsed = time.time() - t0
    print(f"\n  Done: {total_chars:,} chars, {failed} failed batch(es), {elapsed:.1f}s")
    print(f"  Next: python pipeline/stitch.py {chunks_dir} -o {out_dir / (stem + '-stitched.md')}")
    return {"chunks_dir": str(chunks_dir), "failed": failed, "chars": total_chars}


def main():
    ap = argparse.ArgumentParser(description="VLM OCR: scanned book PDF -> markdown page chunks via Gemini.")
    ap.add_argument("pdf", help="Path to the scanned PDF")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Page render DPI (default: {DEFAULT_DPI})")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                    help="Pages per API call (default: 1 — larger batches fail silently)")
    ap.add_argument("--output-dir", default=None, help="Output dir (default: <pdf-dir>/<stem>-ocr/)")
    ap.add_argument("--first-page", type=int, default=1, help="1-based first page to OCR")
    ap.add_argument("--last-page", type=int, default=0, help="1-based last page to OCR (0 = end)")
    args = ap.parse_args()

    process_pdf(args.pdf, args.output_dir, args.model, args.dpi, args.batch,
                args.first_page, args.last_page)


if __name__ == "__main__":
    main()
