import os
import re
import logging
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF  ->  pip install pymupdf
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Heuristics: headers / back-matter
# ------------------------------------------------------------

HEADER_PATTERNS = [
    r"^\d+\s+TALES AND\s*LEGENDS",
    r"^TALES AND\s*LEGENDS\b",
    r"^ANIMAL TALES\b",
    r"^PROVERBS ANDFOLK SAYINGS\b",
    r"^PROVERBS ANDRIDDLES\b",
    r"^FOLKQUIPS\b",
    r"^RIDDLES\b",
    r"^NOTES\b",
    r"^GLOSSARY\b",
    r"^INDEX\b",
]

_HEADER_REGEXES = [re.compile(p, re.IGNORECASE) for p in HEADER_PATTERNS]


def looks_like_running_header(line: str) -> bool:
    """Detect page headers / section labels we don't want as story titles."""
    s = line.strip()
    if not s:
        return False

    # Explicit known headers
    if any(rx.match(s) for rx in _HEADER_REGEXES):
        return True

    # All caps + trailing page number (very generic book header pattern)
    if re.search(r"[a-z]", s) is None and re.search(r"\b\d{2,4}\b$", s):
        return True

    return False


BACK_MATTER_MARKERS = [
    r"\nNOTES\b",
    r"\nGLOSSARY\b",
    r"\nINDEX\b",
    r"\nTheJewish Bookshelf\b",
    r"\nHerearetheBooks thatExplore\b",
]


def strip_back_matter(text: str) -> str:
    """
    Remove NOTES / GLOSSARY / INDEX / publisher ads from the tail of the book.

    Safer heuristic:
    - For each marker, look for the **last** occurrence.
    - Only treat it as back-matter if it is in the **back half** of the text.
    - Cut at the earliest such 'last occurrence' among all markers.
    """
    n = len(text)
    cutoff = n

    for pat in BACK_MATTER_MARKERS:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if not matches:
            continue

        last = matches[-1]
        pos = last.start()

        # Only consider this "back matter" if it appears in the back half of the book
        if pos > n * 0.5 and pos < cutoff:
            cutoff = pos

    if cutoff < n:
        logger.info(
            "Stripping back matter starting at char %d of %d (~%.1f%% of book)",
            cutoff,
            n,
            100 * cutoff / max(n, 1),
        )
        return text[:cutoff]

    return text


# ------------------------------------------------------------
# PDF loading / cleaning with PyMuPDF (fitz)
# ------------------------------------------------------------

def extract_page_text_fitz(page: "fitz.Page") -> str:
    """
    Extract text from a page using word coordinates and rebuild lines
    with proper spacing.
    """
    # words: [x0, y0, x1, y1, "text", block_no, line_no, word_no]
    words = page.get_text("words")
    if not words:
        return ""

    # Group words by (block_no, line_no)
    lines_by_key = {}
    for x0, y0, x1, y1, txt, block_no, line_no, word_no in words:
        key = (block_no, line_no)
        lines_by_key.setdefault(key, []).append((x0, txt))

    # Sort lines in reading order: by block, then line
    sorted_lines = sorted(lines_by_key.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    text_lines = []
    for (_block, _line), items in sorted_lines:
        # sort words left-to-right
        items.sort(key=lambda t: t[0])
        words_only = [w for _, w in items]

        # join words with a space; this is what fixes the smashed-together problem
        line_text = " ".join(words_only).strip()
        if line_text:
            text_lines.append(line_text)

    # join lines with newline
    return "\n".join(text_lines)


def extract_pdf_text_fitz(pdf_path: str) -> str:
    """
    Read a whole PDF with PyMuPDF and return raw text with
    reasonable line breaks and spaces.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        page_text = extract_page_text_fitz(page)
        if page_text:
            pages_text.append(page_text)

    # Separate pages with a blank line so your splitter can see boundaries
    full_text = "\n\n".join(pages_text)
    return full_text.strip()


def clean_pdf_text(text: str) -> str:
    """
    Clean raw text but **preserve newlines** so that:
      - story titles can be inferred from first lines
      - text splitter separators on '\n\n' etc. still work
    """
    if not text:
        return ""

    # Normalize spaces / tabs inside lines
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize crazy sequences of blank lines
    # 3+ newlines -> exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing spaces on each line
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)

    # Final trim
    return cleaned.strip()


def load_pdfs_from_directory(pdf_dir: str) -> List[Tuple[str, str]]:
    """
    Return list of (file_path, cleaned_text) for every PDF in the directory,
    using PyMuPDF (fitz) for extraction.
    If one PDF fails, log and continue (don't bail out).
    """
    pdf_dir_path = Path(pdf_dir)
    if not pdf_dir_path.exists():
        logger.error("PDF directory '%s' does not exist.", pdf_dir)
        return []

    pdf_files = sorted(pdf_dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in '%s'.", pdf_dir)
        return []

    results: List[Tuple[str, str]] = []

    logger.info("Found %d PDF files in '%s'", len(pdf_files), pdf_dir)

    for pdf_path in pdf_files:
        try:
            logger.info("Reading (fitz) %s", pdf_path.name)
            raw_text = extract_pdf_text_fitz(str(pdf_path))
            cleaned_text = clean_pdf_text(raw_text)
            cleaned_text = strip_back_matter(cleaned_text)

            if cleaned_text:
                results.append((str(pdf_path), cleaned_text))
            else:
                logger.warning("No extractable text in '%s'", pdf_path.name)

        except Exception as e:  # noqa: BLE001
            logger.error("Error loading '%s' with fitz: %s", pdf_path.name, e)

    logger.info("Successfully loaded text from %d/%d PDFs", len(results), len(pdf_files))
    return results


# ------------------------------------------------------------
# Chunking into story-ish segments
# ------------------------------------------------------------

def _guess_book_name_from_path(path_str: str) -> str:
    """Best-effort guess a human-ish book name from file name."""
    stem = Path(path_str).stem
    stem = stem.replace("_", " ").replace("-", " ")
    return stem.title()


def _infer_story_title(chunk_text: str, book_name: str, index: int) -> str:
    """
    Heuristic:
    - Scan top non-empty lines, skipping obvious headers
    - If a line is short, capitalized, and not clearly mid-sentence, treat as title
    - Otherwise fall back to 'Book Name – Story segment N'
    """
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    if not lines:
        return f"{book_name} – Story segment {index + 1}"

    for candidate in lines:
        # skip page headers / section labels
        if looks_like_running_header(candidate):
            continue

        # too long → probably body text, not a title
        if len(candidate) > 80:
            continue

        # avoid obvious mid-sentence stuff
        if candidate[0].islower():
            continue
        if candidate.endswith((",", ";")):
            continue

        # if it has a comma in the middle but doesn't look like a clean phrase, skip
        if "," in candidate and not candidate.endswith((".", "!", "?", ":", ";", ",")):
            continue

        return candidate

    # fallback
    return f"{book_name} – Story segment {index + 1}"


def build_story_documents(pdf_texts: List[Tuple[str, str]]) -> List[Document]:
    """
    Turn (file_path, text) pairs into a list of LangChain Documents
    with useful metadata: book_name, story_title, source_file, etc.
    Includes verbose logging so you can see chunking in real time.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
    )

    documents: List[Document] = []
    chunk_counter = 0

    for file_path, text in pdf_texts:
        book_name = _guess_book_name_from_path(file_path)

        chunks = splitter.split_text(text)
        logger.info(
            "Split '%s' into %d chunks", Path(file_path).name, len(chunks)
        )

        for idx, chunk in enumerate(chunks):
            story_title = _infer_story_title(chunk, book_name, idx)

            # DEBUG: show chunk content and title guess
            logger.info(
                "  [Chunk %d] %d chars | title guess: %r",
                chunk_counter,
                len(chunk),
                story_title,
            )
            logger.info("-" * 40)
            preview = chunk[:500].replace("\n", " ")
            logger.info("%s", preview)
            logger.info("-" * 40)

            metadata = {
                "source_file": str(Path(file_path).name),
                "book_name": book_name,
                "story_title": story_title,
                # page is approximate / unknown at this level
                "page": None,
                "chunk_id": idx,
            }

            documents.append(Document(page_content=chunk, metadata=metadata))
            chunk_counter += 1

    logger.info("Built %d total chunks from %d PDFs", len(documents), len(pdf_texts))
    return documents


# ------------------------------------------------------------
# Vector DB creation
# ------------------------------------------------------------

def create_vector_database(
    pdf_dir: str = "story_pdfs",
    persist_directory: str = "chroma_db",
    collection_name: str = "story_collection",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 100,
) -> Chroma:
    """
    End-to-end:
    - load and clean PDFs (with fitz)
    - strip back-matter (safely)
    - chunk into Documents (with debug logging)
    - embed and persist to a Chroma collection
    """
    logger.info("Starting PDF ingestion from '%s'", pdf_dir)
    pdf_texts = load_pdfs_from_directory(pdf_dir)
    if not pdf_texts:
        raise RuntimeError("No usable PDFs found; aborting vector DB creation.")

    documents = build_story_documents(pdf_texts)

    # Single, explicit embedding model config (no double-loading).
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Wipe and recreate persist directory to avoid stale / duplicate data.
    persist_path = Path(persist_directory)
    if persist_path.exists():
        logger.info("Clearing existing Chroma directory '%s'", persist_directory)
        for child in persist_path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                for root, dirs, files in os.walk(child, topdown=False):
                    for name in files:
                        Path(root, name).unlink()
                    for name in dirs:
                        Path(root, name).rmdir()
                child.rmdir()
    else:
        persist_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Creating Chroma collection '%s' at '%s'",
        collection_name,
        persist_directory,
    )

    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    # Proper batching: we only embed + add each chunk once
    for start in tqdm(
        range(0, len(documents), batch_size),
        desc="Adding documents to Chroma",
    ):
        batch = documents[start:start + batch_size]
        vectordb.add_documents(batch)

    vectordb.persist()
    logger.info("Vector DB created with %d documents.", len(documents))
    return vectordb


if __name__ == "__main__":
    create_vector_database()
