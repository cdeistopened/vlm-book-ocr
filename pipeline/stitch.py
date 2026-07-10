#!/usr/bin/env python3
"""Stitch per-page VLM transcriptions into one flowing manuscript.

Input: a directory of *.md chunk files containing blocks:
    === PAGE 039 ===
    (markdown transcription, possibly ending with a hyphenated word)

Joins, in order:
  1. page ends with "-"           -> join the hyphenated word with the next
                                     page's first word (no space, drop hyphen)
  2. page ends mid-sentence       -> merge last paragraph with next page's
     (no terminal punctuation)       first paragraph (single space)
     and next page starts lowercase/continuation
  3. otherwise                    -> paragraph break between pages

FOOTNOTES blocks are lifted out per page and appended at the end under
"## Notes (per source page)" with their source page number — renumbering
and re-anchoring is a later editorial pass, not the stitcher's job.

Usage:
    python pipeline/stitch.py <chunks_dir> -o stitched.md
"""
import argparse
import re
import sys
from pathlib import Path

TERMINAL = re.compile(r'[.!?:;”’"\')\]]$')


def load_pages(chunks_dir: Path):
    pages = {}
    for f in sorted(chunks_dir.glob("*.md")):
        text = f.read_text()
        for m in re.finditer(r'=== PAGE (\d+) ===\n(.*?)(?=\n=== PAGE |\Z)', text, re.S):
            pages[int(m.group(1))] = m.group(2).strip()
    return pages


def split_notes(body):
    m = re.search(r'\nFOOTNOTES:\s*\n(.*)\Z', body, re.S)
    if not m:
        return body.strip(), None
    return body[:m.start()].strip(), m.group(1).strip()


def stitch(pages):
    flow, notes = [], []
    for num in sorted(pages):
        body, fn = split_notes(pages[num])
        if fn:
            notes.append((num, fn))
        if not body:
            continue
        paras = [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]
        if flow and paras:
            prev = flow[-1]
            head = paras[0]
            starts_heading = head.startswith('#')
            if prev.endswith('-') and not starts_heading:
                # cross-page hyphenated word
                flow[-1] = prev[:-1] + head.split(' ', 1)[0] + (
                    ' ' + head.split(' ', 1)[1] if ' ' in head else '')
                paras = paras[1:]
            elif (not TERMINAL.search(prev) and not prev.endswith('#')
                  and not starts_heading and head[:1].islower()):
                # sentence continues across the page turn
                flow[-1] = prev + ' ' + head
                paras = paras[1:]
        flow.extend(paras)
    out = '\n\n'.join(flow)
    if notes:
        out += '\n\n## Notes (per source page)\n\n'
        out += '\n\n'.join(f'**p. {n}** {t}' for n, t in notes)
    return out


def main():
    ap = argparse.ArgumentParser(description="Stitch === PAGE NNN === chunks into one manuscript.")
    ap.add_argument("chunks_dir", help="Directory of *.md chunk files with === PAGE NNN === blocks")
    ap.add_argument("-o", "--output", default="stitched.md", help="Output file (default: stitched.md)")
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.is_dir():
        sys.exit(f"not a directory: {chunks_dir}")
    pages = load_pages(chunks_dir)
    if not pages:
        sys.exit(f"no '=== PAGE NNN ===' blocks found in {chunks_dir}/*.md")

    out = Path(args.output)
    out.write_text(stitch(pages) + '\n')
    print(f"stitched {len(pages)} pages -> {out} "
          f"({len(out.read_text().split())} words)")


if __name__ == '__main__':
    main()
