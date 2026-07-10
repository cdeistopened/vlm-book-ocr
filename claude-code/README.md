# Running the pipeline with Claude Code (no API key, no per-token bill)

The same pipeline works with Claude Code driving Sonnet subagents instead of the Gemini API. This is the path to use when Gemini blocks a book with `RECITATION` (any book Google has in its corpus), and it's effectively free if you already have a Claude subscription — Claude Code usage is covered by the plan, so there is no per-token bill.

## 1. Extract pages as images

```bash
mkdir -p pages
pdftoppm -jpeg -r 300 book.pdf pages/page
```

(`pdftoppm` ships with poppler: `brew install poppler` / `apt install poppler-utils`.)

You now have `pages/page-001.jpg`, `pages/page-002.jpg`, …

## 2. Fan out transcription subagents

Open Claude Code in the repo root and ask it to transcribe the pages in parallel, 4 pages per subagent. A prompt that works:

> Transcribe the scanned book pages in `pages/` to markdown. Launch Sonnet subagents in parallel, 4 pages per agent. Each agent gets the prompt template below. Have each agent write its output to `chunks/pages-NNN-NNN.md`. When all agents finish, verify every page number appears exactly once across the chunk files.

The per-agent prompt template (this is the important part — keep it exact):

```
Read pipeline/TRANSCRIBE-RULES.md and follow every rule exactly.

Transcribe these 4 scanned book pages to markdown:
- pages/page-037.jpg  (page 37)
- pages/page-038.jpg  (page 38)
- pages/page-039.jpg  (page 39)
- pages/page-040.jpg  (page 40)

Write the result to chunks/pages-037-040.md in exactly this format,
one block per page:

=== PAGE 037 ===
(transcription of page 37)

=== PAGE 038 ===
(transcription of page 38)

...

No commentary, no code fences around the output — just the page blocks.
```

Why 4 pages per agent: enough to amortize the agent spin-up, small enough that a bad read is cheap to redo and the agent never drifts. One page per API call is the rule on the Gemini path; on the Claude path the agent reads each image as a separate file so 4 works fine.

## 3. Stitch

```bash
python pipeline/stitch.py chunks/ -o book-stitched.md
```

The stitcher joins hyphenated words across page turns, merges mid-sentence page breaks, and collects `FOOTNOTES:` blocks at the end with their source page numbers.

## Spot-checking

Pick 3–4 random pages and diff the transcription against the scan by eye. Rule 5 (never invent headings) and rule 1 (verbatim, no modernizing) are where models most want to "help."
