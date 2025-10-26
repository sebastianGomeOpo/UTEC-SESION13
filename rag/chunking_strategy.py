from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_semantic_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Returns a configured instance of RecursiveCharacterTextSplitter for semantic chunking.

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter instance.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],  # Standard hierarchical separators
        chunk_size=1000,                    # Size of chunks
        chunk_overlap=200,                  # Overlap between chunks
        length_function=len                 # Use standard length function
    )
    return text_splitter
