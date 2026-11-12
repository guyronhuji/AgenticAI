"""Retrieval, three ways (Week 10, segment 2.3.6).

  1. classic RAG:   chunk -> embed -> top-k by cosine -> stuff into the prompt
  2. hybrid:        BM25 (exact words) + vectors (meaning), fused by RRF
  3. agentic:       retrieval exposed as a search() TOOL inside the agent loop,
                    so the model can re-query when the first hits are wrong

The vector store is a numpy matrix. A "vector database" is this plus
persistence and scale; the hard parts are chunking and query formulation.

Embeddings: sentence-transformers if installed (all-MiniLM-L6-v2, ~90 MB),
else Ollama's embedding endpoint (nomic-embed-text), else a hashed
bag-of-words fallback so tests run anywhere (EMBED_BACKEND=hash).
"""
from __future__ import annotations

import math
import os
import pathlib
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "auto")  # auto|st|ollama|hash
EMBED_MODEL = os.environ.get("EMBED_MODEL", "")


# ----------------------------------------------------------- documents --
@dataclass
class Chunk:
    id: str
    doc: str
    text: str          # what gets embedded and shown to the model


def load_corpus(folder) -> dict[str, str]:
    """{doc_id: text} from a folder of .txt/.md files (first line = title)."""
    folder = pathlib.Path(folder)
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.md"))}


def chunk_text(doc_id, text, max_words=120, overlap=20, contextual=True) -> list[Chunk]:
    """Paragraph-aware chunks of ~max_words with overlap. With `contextual`,
    every chunk is prefixed by the document title so a chunk read in isolation
    still says what it is about (the 'contextual retrieval' trick)."""
    title = text.strip().splitlines()[0].strip() if text.strip() else doc_id
    words = text.split()
    chunks, i, n = [], 0, 0
    while i < len(words):
        piece = " ".join(words[i:i + max_words])
        prefix = f"[{title}] " if contextual and not piece.startswith(title) else ""
        chunks.append(Chunk(id=f"{doc_id}#{n}", doc=doc_id, text=prefix + piece))
        n += 1
        if i + max_words >= len(words):
            break
        i += max_words - overlap
    return chunks


# ------------------------------------------------------------ embeddings --
_st_model = None


def _embed_st(texts):
    global _st_model
    from sentence_transformers import SentenceTransformer
    if _st_model is None:
        _st_model = SentenceTransformer(EMBED_MODEL or "all-MiniLM-L6-v2")
    return np.asarray(_st_model.encode(texts, normalize_embeddings=True))


def _embed_ollama(texts):
    import ollama
    from common.llm import BASE_URL
    r = ollama.Client(host=BASE_URL).embed(model=EMBED_MODEL or "nomic-embed-text", input=list(texts))
    v = np.asarray(r.embeddings, dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _embed_hash(texts, dim=512):
    """Deterministic hashed bag-of-words: no semantics, but the pipeline runs."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in _tokens(t):
            out[i, hash(w) % dim] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def embed(texts):
    texts = list(texts)
    backend = EMBED_BACKEND
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
            backend = "st"
        except ImportError:
            backend = "ollama"
    if backend == "st":
        return _embed_st(texts)
    if backend == "ollama":
        try:
            return _embed_ollama(texts)
        except Exception:
            return _embed_hash(texts)
    return _embed_hash(texts)


# ------------------------------------------------------------------ BM25 --
def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """The 25-line version of the classic keyword scorer (k1=1.5, b=0.75)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [_tokens(t) for t in texts]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.df = Counter(w for d in self.docs for w in set(d))
        self.N = len(self.docs)

    def scores(self, query):
        q = _tokens(query)
        out = np.zeros(self.N)
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            for w in q:
                if w not in tf:
                    continue
                idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                denom = tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avg)
                out[i] += idf * tf[w] * (self.k1 + 1) / denom
        return out


# ----------------------------------------------------------------- index --
class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectors = embed([c.text for c in chunks])
        self.bm25 = BM25([c.text for c in chunks])

    @classmethod
    def from_folder(cls, folder, **chunk_kwargs):
        chunks = []
        for doc_id, text in load_corpus(folder).items():
            chunks += chunk_text(doc_id, text, **chunk_kwargs)
        return cls(chunks)

    def search_vector(self, query, k=3):
        q = embed([query])[0]
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    def search_bm25(self, query, k=3):
        s = self.bm25.scores(query)
        top = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in top if s[i] > 0]

    def search_hybrid(self, query, k=3, pool=10, rrf_k=60):
        """Reciprocal-rank fusion of the two rankings: score = sum 1/(rrf_k+rank)."""
        fused = Counter()
        for ranking in (self.search_vector(query, pool), self.search_bm25(query, pool)):
            for rank, (chunk, _) in enumerate(ranking, 1):
                fused[chunk.id] += 1.0 / (rrf_k + rank)
        by_id = {c.id: c for c in self.chunks}
        return [(by_id[cid], score) for cid, score in fused.most_common(k)]

    def search(self, query, k=3, mode="hybrid"):
        return {"vector": self.search_vector, "bm25": self.search_bm25,
                "hybrid": self.search_hybrid}[mode](query, k)


def format_hits(hits):
    return "\n\n".join(f"[{c.id}] {c.text}" for c, _ in hits) or "(no results)"


def make_search_tool(index: Index, k=3, mode="hybrid"):
    """Retrieval as a TOOL for the agent loop (agentic search)."""
    def search_corpus(query: str) -> str:
        """Search the document corpus. Returns the most relevant passages with their ids. Call again with different words if the first results do not contain what you need.

        query: search terms or a question
        """
        return format_hits(index.search(query, k=k, mode=mode))
    return search_corpus


RAG_PROMPT = """Answer the question using ONLY the sources below. Cite the source id \
in brackets after each claim, like [doc#0]. If the sources do not contain the \
answer, say exactly: "The sources do not contain this."

Sources:
{sources}

Question: {question}"""
