"""Load the document corpus and split it into retrieval-sized chunks.

Handles PDF, Markdown and plain text out of data/documents (recursively), then
applies a structure-aware recursive split. Every chunk carries enough metadata
to cite it back to a filename and page.

Run standalone for a sanity check:

    python -m src.ingestion.loader
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Allow `python -m src.ingestion.loader` and `python src/ingestion/loader.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

PDF_SUFFIXES = {".pdf"}
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | TEXT_SUFFIXES


# --- Token counting --------------------------------------------------------
def build_token_counter() -> Callable[[str], int]:
    """Return a function that counts tokens in a string.

    Uses tiktoken's cl100k_base when available. That is not the exact
    vocabulary nomic-embed-text uses, but chunk sizing only needs to be
    consistent and in the right ballpark — being off by a few percent on token
    counts costs nothing, whereas splitting by raw characters silently
    produces chunks of wildly varying information density (a table of numbers
    packs ~2x the tokens per character that prose does).

    Falls back to a 4-characters-per-token heuristic if tiktoken is missing so
    the pipeline still runs.
    """
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(encoding.encode(text, disallowed_special=()))

        return count
    except Exception:  # noqa: BLE001 - any tiktoken failure degrades gracefully
        def count(text: str) -> int:
            return max(1, len(text) // 4)

        return count


# --- Loading ---------------------------------------------------------------
def discover_files(documents_dir: Path | None = None) -> list[Path]:
    """List every supported source file under the documents directory."""
    documents_dir = documents_dir or config.DOCUMENTS_DIR
    if not documents_dir.exists():
        return []
    return sorted(
        p
        for p in documents_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _pypdf_loader_cls():
    """Resolve PyPDFLoader, preferring a standalone package if one exists.

    langchain-community is being sunset, but as of writing no standalone
    langchain-pypdf package has been published, so community remains the
    supported home for this loader. The warning is suppressed because it is
    not actionable yet; revisit when the split lands.
    """
    try:
        from langchain_pypdf import PyPDFLoader  # type: ignore[import-not-found]

        return PyPDFLoader
    except ImportError:
        pass

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader


def _load_pdf(path: Path) -> list[Document]:
    # PyPDFLoader emits one Document per page, which is what gives us real
    # page numbers for citation. Splitting happens later, within each page.
    return _pypdf_loader_cls()(str(path)).load()


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={})]


def load_documents(documents_dir: Path | None = None) -> list[Document]:
    """Load every supported file into page-level Documents with clean metadata.

    Metadata written here:
      source      - bare filename, what a citation should show
      source_path - path relative to data/documents, disambiguates duplicates
      file_type   - "pdf" | "markdown" | "text"
      page        - 1-based page number, or config.NO_PAGE for unpaginated files
    """
    documents_dir = documents_dir or config.DOCUMENTS_DIR
    files = discover_files(documents_dir)

    documents: list[Document] = []
    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix in PDF_SUFFIXES:
                pages = _load_pdf(path)
                file_type = "pdf"
            else:
                pages = _load_text(path)
                file_type = "markdown" if suffix in {".md", ".markdown"} else "text"
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort ingest
            print(f"  ! skipped {path.name}: {type(exc).__name__}: {exc}")
            continue

        relative = path.relative_to(documents_dir).as_posix()
        for page_doc in pages:
            # PyPDFLoader's `page` is 0-based; report it 1-based to match what
            # a reader sees in a PDF viewer.
            raw_page = page_doc.metadata.get("page")
            page = int(raw_page) + 1 if isinstance(raw_page, int) else config.NO_PAGE

            page_doc.metadata = {
                "source": path.name,
                "source_path": relative,
                "file_type": file_type,
                "page": page,
            }
            documents.append(page_doc)

    return documents


# --- Chunking --------------------------------------------------------------
def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Construct the corpus splitter.

    Why RecursiveCharacterTextSplitter over a naive fixed-size split:

    A fixed-size splitter cuts every N characters regardless of what is at that
    offset, so it routinely severs a sentence mid-clause or detaches a figure
    from the paragraph that explains it. The resulting chunk embeds to a point
    that represents no coherent idea, which is precisely the failure mode that
    makes a RAG system retrieve plausible-looking but useless context.

    The recursive splitter instead walks a ladder of separators from coarsest
    to finest, and only descends when a piece is still too large. The ladder
    below is ordered by how much semantic boundary each separator implies:
    markdown headings first (a new section is the strongest possible break),
    then paragraphs, then lines, then sentences, and only as a last resort
    words and characters. In practice most chunks land on a paragraph boundary
    and stay internally coherent.

    Sizing:
      800 tokens - large enough to hold a full argument (a finding plus its
        supporting numbers, or a complete subsection of a report), which is
        what business-research questions actually need. Much smaller and
        answers get stitched from fragments that each lack context; much
        larger and the embedding averages several topics together, blurring
        the vector and hurting retrieval precision. It also keeps ~8-10 chunks
        comfortably inside a typical context window at answer time.
      100 tokens overlap (12.5%) - carries the tail of each chunk into the
        next so a fact stated at a boundary is retrievable from either side.
        Enough to preserve a sentence or two of lead-in; small enough that
        index size and duplicate hits stay manageable.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE_TOKENS
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP_TOKENS

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Sizes are in tokens because that is the unit both the embedding model
        # and the downstream LLM budget in.
        length_function=build_token_counter(),
        separators=[
            "\n## ",    # markdown section
            "\n### ",   # markdown subsection
            "\n\n",     # paragraph
            "\n",       # line
            ". ",       # sentence
            " ",        # word
            "",         # character (last resort)
        ],
        keep_separator=True,
    )


def chunk_documents(
    documents: Iterable[Document],
    splitter: RecursiveCharacterTextSplitter | None = None,
) -> list[Document]:
    """Split page-level Documents into chunks, adding chunk_index metadata.

    chunk_index counts within a source file, so (source, chunk_index) uniquely
    identifies a chunk and reads naturally in a citation.
    """
    splitter = splitter or build_splitter()
    chunks = splitter.split_documents(list(documents))

    per_source: dict[str, int] = {}
    for chunk in chunks:
        key = chunk.metadata.get("source_path", chunk.metadata.get("source", "?"))
        index = per_source.get(key, 0)
        chunk.metadata["chunk_index"] = index
        per_source[key] = index + 1

    return chunks


# --- Sanity check ----------------------------------------------------------
def preview_chunks(chunks: list[Document], n: int = 3, width: int = 400) -> None:
    """Print a few chunks so chunk quality can be eyeballed."""
    if not chunks:
        print("No chunks to preview.")
        return

    # Sample across the corpus rather than taking the first n, which would all
    # come from the same file and usually be title pages.
    if len(chunks) <= n:
        picks = list(range(len(chunks)))
    else:
        step = len(chunks) / n
        picks = [int(i * step) for i in range(n)]

    counter = build_token_counter()
    for i in picks:
        chunk = chunks[i]
        meta = chunk.metadata
        page = meta.get("page", config.NO_PAGE)
        page_label = f"p.{page}" if page != config.NO_PAGE else "no pages"
        body = chunk.page_content.strip()
        truncated = body[:width] + ("…" if len(body) > width else "")

        print(f"\n--- chunk {i} | {meta.get('source', '?')} | {page_label} "
              f"| chunk_index={meta.get('chunk_index')} "
              f"| {counter(chunk.page_content)} tokens ---")
        print(truncated)


def load_and_chunk(documents_dir: Path | None = None) -> tuple[list[Document], list[Document]]:
    """Convenience wrapper returning (page_documents, chunks)."""
    documents = load_documents(documents_dir)
    chunks = chunk_documents(documents)
    return documents, chunks


def main() -> int:
    config.enable_utf8_stdout()
    print(f"Scanning {config.DOCUMENTS_DIR} …")
    documents, chunks = load_and_chunk()

    if not documents:
        print(
            "\nNo supported documents found."
            f"\nDrop .pdf, .md or .txt files into {config.DOCUMENTS_DIR} and rerun."
        )
        return 1

    sources = {d.metadata["source"] for d in documents}
    print(f"\nLoaded {len(documents)} pages from {len(sources)} file(s)")
    print(f"Produced {len(chunks)} chunks")

    preview_chunks(chunks, n=3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
