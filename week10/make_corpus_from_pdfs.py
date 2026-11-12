"""Homework helper: turn a folder of PDFs into the text corpus the retrieval
code expects (one .txt per paper, first line = title).

    uv run python week10/make_corpus_from_pdfs.py ~/papers my_corpus
    uv run python week10/06_retrieval_demo.py ...        # then point Index.from_folder at my_corpus

The first line of each .txt is the title the chunker prefixes to every chunk,
so fix it by hand if the PDF's first line is a journal header. Text extraction
from PDFs is lossy (two-column layouts interleave, equations vanish); that is
real data, and part of homework 2 is noticing what it did to your retrieval.
"""
import pathlib
import re
import sys


def pdf_to_text(path):
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text


def guess_title(text, fallback):
    for line in text.splitlines():
        line = line.strip()
        if 15 <= len(line) <= 200 and not re.search(r"arxiv|doi|journal|vol\.|©|copyright", line, re.I):
            return line
    return fallback


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = pathlib.Path(sys.argv[1]).expanduser(), pathlib.Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(src.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs in {src}")
    for pdf in pdfs:
        text = pdf_to_text(pdf)
        title = guess_title(text, pdf.stem)
        out = dst / (re.sub(r"[^a-z0-9]+", "_", pdf.stem.lower()).strip("_") + ".txt")
        out.write_text(title + "\n" + text, encoding="utf-8")
        print(f"{pdf.name:50s} -> {out.name}  ({len(text.split())} words)  title: {title[:60]}")
    print(f"\n{len(pdfs)} files written to {dst}/. Check the first line of each; it becomes the chunk prefix.")
