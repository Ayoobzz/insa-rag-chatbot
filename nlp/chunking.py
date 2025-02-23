from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_data(documents, chunk_size=250, overlap=20):
    """Chunk the given documents into smaller pieces."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunked_docs = []
    for url, content in documents.items():
        chunks = splitter.split_text(content[0])

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "metadata": {
                    "url": url,
                    "title": content[1],
                    "chunk_number": i + 1
                }
            })
    return chunked_docs

