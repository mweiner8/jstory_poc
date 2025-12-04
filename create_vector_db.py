"""
Script to process story PDFs and create a ChromaDB vector database.
Run this once to set up your database.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from tqdm import tqdm
import json

# Configuration
PDF_DIRECTORY = "./story_pdfs"  # Put your PDF files here
CHROMA_DB_DIR = "./chroma_db"   # Vector database will be saved here
COLLECTION_NAME = "story_collection"

# Story metadata file (optional - for better organization)
METADATA_FILE = "./story_metadata.json"


def clean_pdf_text(text: str) -> str:
    """Clean up PDF extraction issues."""
    import re

    # Fix missing spaces between words (common PDF issue)
    # Insert space before capital letters in middle of "words"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Fix missing spaces after punctuation
    text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def load_pdfs_from_directory(directory: str) -> List:
    """Load all PDFs from a directory."""
    documents = []

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        print(f"Please add your PDF files to {directory} and run again.")
        return documents

    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return documents

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        print(f"Loading: {pdf_file}")

        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                doc.page_content = clean_pdf_text(doc.page_content)
                doc.metadata['source_file'] = pdf_file
                doc.metadata['book_name'] = pdf_file.replace('.pdf', '')

            documents.extend(docs)
            print(f"  Loaded {len(docs)} pages from {pdf_file}")
        except Exception as e:
            print(f"  Error loading {pdf_file}: {e}")

    return documents


def extract_story_metadata(doc):
    """Extract story titles and improve metadata."""
    content = doc.page_content.strip()
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    if not lines:
        return "Untitled"

    # Look for title patterns in first few lines
    for line in lines[:5]:
        # Skip page numbers, headers, footers
        if line.isdigit() or len(line) < 10:
            continue

        # Skip lines that are all caps (likely headers)
        if line.isupper() and len(line) > 20:
            continue

        # Good title candidate: reasonable length, starts with capital
        if 10 < len(line) < 150 and line[0].isupper():
            # Check if it looks like a sentence vs title
            if not line.endswith('.') or len(line) < 50:
                return line

    # Fallback: return first non-empty line, truncated
    return lines[0][:100] if lines else "Untitled"


def split_documents_into_stories(documents: List, metadata_file: str = None) -> List:
    """
    Split documents into story-sized chunks.

    If you have a metadata JSON file with story boundaries, use it.
    Otherwise, use intelligent text splitting.
    """

    # Option 1: If you have story metadata (recommended)
    if metadata_file and os.path.exists(metadata_file):
        print("Using metadata file for story boundaries")
        with open(metadata_file, 'r') as f:
            story_metadata = json.load(f)
        # Implementation depends on your metadata structure
        # This is a placeholder - customize based on your needs

    # Option 2: Intelligent text splitting (default)
    print("Splitting documents into story-sized chunks")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    split_docs = text_splitter.split_documents(documents)

    # Add chunk metadata
    for i, doc in enumerate(split_docs):
        doc.metadata['chunk_id'] = i
        if 'story_title' not in doc.metadata:
            doc.metadata['story_title'] = extract_story_metadata(doc)

    return split_docs


def create_vector_database(documents: List, persist_directory: str, collection_name: str):
    """Create and persist ChromaDB vector database."""

    print(f"\nCreating embeddings for {len(documents)} document chunks...")
    print("This may take a few minutes depending on the corpus size...")

    # Use HuggingFace embeddings with batch processing
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )

    # Process in batches to show progress
    batch_size = 100
    print(f"Processing in batches of {batch_size}...")

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing chunks {i + 1}-{min(i + batch_size, len(documents))} of {len(documents)}")

        if i == 0:
            # Create initial database
            print("Generating embeddings with progress tracking...")
            vectordb = Chroma.from_documents(
                documents=tqdm(documents, desc="Embedding chunks"),
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
        else:
            # Add to existing database
            vectordb.add_documents(batch)

    print(f"\nVector database created successfully!")
    print(f"Location: {persist_directory}")
    print(f"Total documents: {len(documents)}")

    return vectordb

def main():
    """Main execution function."""

    print("=" * 60)
    print("Story Collection Vector Database Creator")
    print("=" * 60)

    # Step 1: Load PDFs
    print("\n[Step 1] Loading PDF files...")
    documents = load_pdfs_from_directory(PDF_DIRECTORY)

    if not documents:
        print("\nNo documents to process. Exiting.")
        return

    print(f"Total pages loaded: {len(documents)}")

    # Step 2: Split into stories
    print("\n[Step 2] Processing documents into story chunks...")
    story_chunks = split_documents_into_stories(documents, METADATA_FILE)

    print(f"Total story chunks created: {len(story_chunks)}")

    # Step 3: Create vector database
    print("\n[Step 3] Creating vector database...")
    vectordb = create_vector_database(
        story_chunks,
        CHROMA_DB_DIR,
        COLLECTION_NAME
    )

    # Test the database
    print("\n[Step 4] Testing database with sample query...")
    test_query = "adventure story"
    results = vectordb.similarity_search(test_query, k=3)

    print(f"\nTest query: '{test_query}'")
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Source: {result.metadata.get('source_file', 'Unknown')}")
        print(f"  Preview: {result.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("Setup complete! You can now run the Streamlit app.")
    print("=" * 60)

if __name__ == "__main__":
    main()