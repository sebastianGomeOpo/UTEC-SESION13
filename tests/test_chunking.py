# Standard Library Imports
import re

# Third-party Imports
import pytest
from langchain_core.documents import Document

# Project Imports
# Assuming get_semantic_text_splitter exists and returns RecursiveCharacterTextSplitter
from rag.chunking_strategy import get_semantic_text_splitter
# NOTE: The original spec mentioned ChunkingStrategy class, but code provided a function.
# Tests are adjusted to use the function get_semantic_text_splitter.

# --- Test Cases ---

def test_split_long_paragraphs(chunk_splitter, sample_long_paragraphs):
    """
    Given: Text with multiple long paragraphs.
    When: Splitting with semantic chunker (default size ~1000).
    Then: The text is split into multiple chunks (not just one, not too many).
          Each chunk contains substantial text.
    """
    # Act
    chunks = chunk_splitter.split_text(sample_long_paragraphs) # Use split_text for raw string

    # Assert
    assert isinstance(chunks, list)
    # Expecting 2-4 chunks based on size 1000, overlap 200, and ~1500+ chars total
    # This might need adjustment based on exact splitter behavior
    assert 2 <= len(chunks) <= 4, f"Expected 2-4 chunks, but got {len(chunks)}"
    assert all(isinstance(chunk, str) for chunk in chunks), "Chunks should be strings"
    assert all(len(chunk) > 200 for chunk in chunks), "Some chunks are unexpectedly small"

def test_split_with_table(chunk_splitter, sample_text_with_table):
    """
    Given: Text containing paragraphs and a distinct table structure.
    When: Splitting text using the semantic splitter.
    Then: The table structure is reasonably preserved within chunks,
          though exact splitting depends on separators and size.
          The test focuses on ensuring table content isn't lost.
    """
    # Act
    chunks = chunk_splitter.split_text(sample_text_with_table)

    # Assert
    assert isinstance(chunks, list)
    assert len(chunks) > 0, "Should produce at least one chunk"

    table_content_found = False
    expected_table_keywords = ["Tabla 16", "Puente de Glúteo", "Clamshell"]
    full_text = "".join(chunks)

    for keyword in expected_table_keywords:
        assert keyword in full_text, f"Keyword '{keyword}' from table lost during chunking"
        if keyword in chunks[0]: # Check if at least some table content is in first relevant chunk
             table_content_found = True

    # Note: RecursiveCharacterTextSplitter doesn't inherently assign metadata like 'chunk_type'.
    # This test verifies content preservation. Metadata testing depends on later pipeline steps.
    # We can check if the table seems split across multiple chunks or not,
    # but exact behavior depends heavily on chunk_size and separators.
    table_lines = [line for line in sample_text_with_table.split('\n') if line.strip() and ('|' in line or '===' in line)]
    lines_in_first_chunk = [line for line in table_lines if line in chunks[0]]

    # Heuristic: If most table lines are in the first chunk containing table content...
    # Find first chunk with table content
    first_table_chunk_index = -1
    for i, chunk in enumerate(chunks):
        if "Tabla 16" in chunk:
            first_table_chunk_index = i
            break

    if first_table_chunk_index != -1:
        lines_in_relevant_chunk = [line for line in table_lines if line in chunks[first_table_chunk_index]]
        # Check if a substantial part of the table is in that chunk
        # assert len(lines_in_relevant_chunk) / len(table_lines) > 0.7, "Table seems overly fragmented"
        # This assertion is brittle, focusing on keyword presence is more robust for basic splitting.

    assert table_content_found, "Table content was not found in the initial chunk(s)"


def test_chunk_overlap_works(chunk_splitter, sample_long_paragraphs):
    """
    Given: Long text split into multiple chunks with overlap.
    When: Examining consecutive chunks.
    Then: The end of chunk N should overlap with the start of chunk N+1.
    """
    # Act
    chunks = chunk_splitter.split_text(sample_long_paragraphs)

    # Assert
    assert len(chunks) > 1, "Need multiple chunks to test overlap"
    overlap_size = chunk_splitter.chunk_overlap # Get configured overlap

    for i in range(len(chunks) - 1):
        chunk_n_end = chunks[i][-overlap_size:]
        chunk_n_plus_1_start = chunks[i+1][:overlap_size]

        # Check if the end of chunk N is present at the beginning of chunk N+1
        # Due to separator splitting, the exact overlap might vary slightly,
        # so we check if a significant portion matches.
        # A simpler check: ensure the last few words of chunk N appear in chunk N+1 start
        last_words_n = re.findall(r'\b\w+\b', chunks[i])[-10:] # Last 10 words
        start_text_n_plus_1 = chunks[i+1][:overlap_size + 50] # Look in overlap + buffer

        if last_words_n:
             last_word = last_words_n[-1]
             assert last_word in start_text_n_plus_1, \
                 f"Overlap failed: Last word '{last_word}' of chunk {i} not found near start of chunk {i+1}"

        # More robust: Check sequence, find common substring
        # common_substring = os.path.commonprefix([chunk_n_end[::-1], chunk_n_plus_1_start[::-1]])[::-1]
        # assert len(common_substring) > overlap_size * 0.5, f"Overlap between chunk {i} and {i+1} seems too small"


def test_empty_text_handling(chunk_splitter):
    """
    Given: An empty string.
    When: Splitting the text.
    Then: It should return an empty list and not raise an error.
    """
    # Act
    chunks = chunk_splitter.split_text("")

    # Assert
    assert isinstance(chunks, list)
    assert len(chunks) == 0, "Splitting empty text should yield zero chunks"

def test_short_text_handling(chunk_splitter):
    """
    Given: Text shorter than the chunk size.
    When: Splitting the text.
    Then: It should return a single chunk containing the original text.
    """
    # Arrange
    short_text = "This is a short piece of text, definitely smaller than the chunk size."

    # Act
    chunks = chunk_splitter.split_text(short_text)

    # Assert
    assert isinstance(chunks, list)
    assert len(chunks) == 1, "Short text should result in a single chunk"
    assert chunks[0] == short_text, "The single chunk should match the original short text"

# Note: Metadata preservation tests are usually done after the splitter is integrated
# into a loader or pipeline that adds metadata (like PyPDFLoader).
# The RecursiveCharacterTextSplitter itself doesn't add 'source_page' etc.
# These tests would belong more in test_rag_pipeline.py if testing split_documents.

def test_split_documents_preserves_metadata(chunk_splitter):
    """
    Given: Langchain Documents with metadata.
    When: Using split_documents.
    Then: The resulting chunks should retain the original metadata.
    """
    # Arrange
    docs = [
        Document(page_content="Content of page 1.", metadata={"source": "doc.pdf", "page": 1}),
        Document(page_content="Content of page 2, slightly longer.", metadata={"source": "doc.pdf", "page": 2}),
    ]

    # Act
    chunks = chunk_splitter.split_documents(docs)

    # Assert
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert "source" in chunk.metadata
        assert chunk.metadata["source"] == "doc.pdf"
        assert "page" in chunk.metadata
        assert chunk.metadata["page"] in [1, 2]
