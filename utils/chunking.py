from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 100):
    # Improvement: Updated defaults to match app usage for consistent chunking (smaller for better RAG recall)
    """
    Split text into overlapping chunks.
    Returns list of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

def chunk_documents(documents: list):
    """
    Chunk multiple documents.
    Returns list of all chunks with metadata.
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc['content'])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                'text': chunk,
                'metadata': {'source': doc['filename'], 'chunk_index': i}
            })
    return all_chunks