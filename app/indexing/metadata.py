from langchain_core.documents import Document

from app.ingestion.chunker import DocumentChunk


def chunk_to_document(chunk: DocumentChunk) -> Document:
    """Convert an ingestion chunk into a LangChain document."""
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.content,
        metadata={
            "source_file": chunk.source_file,
            "source_type": chunk.source_type,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
        },
    )


def chunks_to_documents(chunks: list[DocumentChunk]) -> list[Document]:
    return [chunk_to_document(chunk) for chunk in chunks]
