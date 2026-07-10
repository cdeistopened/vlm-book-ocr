# vlm-book-ocr

OCR for old books and letterpress magazines using a vision-language model (Gemini or Claude) instead of a classical OCR engine, plus a deterministic stitcher that joins per-page output into one flowing manuscript.

Two reasons a VLM beats Tesseract on this class of material:

1. **It reads display type and bad scans Tesseract can't.** Real example: on a 1945 letterpress magazine masthead, Tesseract 5.5.2 rendered "OUR DAILY BREAD" as `@OR DANY @884b`. A VLM reads it correctly — old letterpress, decorative faces, foxed paper, two-column layouts all included.
2. **It outputs structured markdown directly** — headings, paragraphs, footnotes, italics — so document structure comes out of the OCR step instead of being reconstructed in a second pass over raw text.

## Two ways to run it

Both produce the same thing: per-page markdown chunks with `=== PAGE NNN ===` markers, which `pipeline/stitch.py` joins into one flowing manuscript (rejoining hyphenated words across page turns, merging mid-sentence breaks, collecting footnotes).

### A. Gemini API (scripted, cheapest)

Requires a `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com) — the free tier is enough for testing. The key is read from the environment; nothing is ever hardcoded.

```bash
pip install google-genai pymupdf
export GEMINI_API_KEY=...

python pipeline/gemini_ocr.py book.pdf --model gemini-3-flash-preview --dpi 300 --batch 1
python pipeline/stitch.py book-ocr/chunks/ -o book-stitched.md
```

The script is resumable — chunks are cached on disk, so re-running after a crash or rate limit picks up where it left off.

### B. Claude Code (no API key, no per-token bill)

Extract pages with `pdftoppm`, then have Claude Code fan out Sonnet subagents (4 pages each) using the same transcription rules, then stitch. Full walkthrough with the exact subagent prompt: [claude-code/README.md](claude-code/README.md).

This is also the fallback when Gemini refuses a book (see gotcha (a) below).

## The transcription rules

[pipeline/TRANSCRIBE-RULES.md](pipeline/TRANSCRIBE-RULES.md) is the contract both engines follow: verbatim (no modernizing, no "fixing" the author), dehyphenate within a page but keep page-boundary hyphens for the stitcher, omit running heads and folios, footnotes in a `FOOTNOTES:` block per page. These rules are the difference between OCR output and a usable manuscript — worth reading before your first run.

## Known gotchas (hard-won)

**a. Gemini RECITATION block.** On books that exist in Google's corpus — anything on Google Books — Gemini refuses transcription with `finishReason=RECITATION`, regardless of public-domain status, after burning thinking tokens. We hit this on Ryan's *A Living Wage* (1906). Claude does not have this failure mode on public-domain works; when Gemini refuses, switch to the Claude Code path.

**b. Thinking models can eat the output budget.** Set `maxOutputTokens` high and `thinkingBudget` low, or the response comes back empty. The script already does this.

**c. Batch small.** One page per call. Multi-page batches fail silently — pages get skipped or merged with no error. `--batch 1` is the default for a reason.

**d. Never let the model infer headings.** A mid-chapter page gets no heading — models will happily invent "Chapter I" from context. This is rule 5 in TRANSCRIBE-RULES.md and it matters more than it sounds.

**e. Full-page VLM reads silently auto-correct printer's errors.** If you need to *detect* print defects (misspellings as printed, broken type), a full-page read will quietly fix them. Crop and zoom on the region instead.

## Cost ballpark

Roughly 1.3K tokens per page in, 500 out:

| Engine | Per page | 374-page book |
|---|---|---|
| Gemini 3 Flash | ~$0.002 | ~$0.80 |
| Sonnet 5 (API) | ~$0.01 | ~$3–4 |
| Claude Code (subscription) | ~$0 marginal | ~$0 marginal |

**Model choice (tested on a 1906 book):** Sonnet is the workhorse — on clean body text Haiku matched it word-for-word, but Haiku mis-transcribed broken type and silently added a French accent that isn't in the print. If per-token cost matters, Haiku is viable for clean scans with a stronger model adjudicating flagged pages; otherwise use Sonnet.

## Layout

```
pipeline/
  gemini_ocr.py        PDF -> page images -> Gemini -> chunks/ (resumable)
  stitch.py            chunks/ -> one stitched manuscript
  TRANSCRIBE-RULES.md  the per-page transcription contract
claude-code/
  README.md            the same pipeline via Claude Code subagents
```

MIT licensed.
