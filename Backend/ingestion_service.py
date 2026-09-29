from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from embedding_service import create_embedding
from qdrant_service import add_document_chunk, create_collection


def ingest_pdf(
    file_path,
    document_name,
    chat_id
):

    print("========================================")
    print("Starting PDF ingestion...")
    print("Document:", document_name)
    print("Chat ID:", chat_id)
    print("========================================")

    # 1. Make sure Qdrant collection exists
    create_collection()

    # 2. Extract text from PDF
    text = extract_text_from_pdf(file_path)

    if not text.strip():

        raise ValueError(
            "No text could be extracted from the PDF."
        )

    print(
        "PDF text extracted successfully"
    )

    # 3. Split text into chunks
    chunks = chunk_text(
        text,
        chunk_size=300,
        overlap=50
    )

    if not chunks:

        raise ValueError(
            "No chunks were created from the PDF."
        )

    print(
        "Total chunks:",
        len(chunks)
    )

    total_chunks = len(chunks)

    # 4. Create embeddings and store chunks
    for i, chunk in enumerate(chunks):

        print(
            f"Processing chunk "
            f"{i + 1}/{total_chunks}..."
        )

        embedding = create_embedding(
            chunk
        )

        add_document_chunk(

            text=chunk,

            embedding=embedding,

            document_name=document_name,

            chunk_number=i + 1,

            total_chunks=total_chunks,

            chat_id=chat_id
        )

        print(
            f"Chunk {i + 1}/{total_chunks} "
            f"stored successfully"
        )

    print("========================================")
    print(
        "PDF ingestion completed successfully"
    )
    print(
        "Document:",
        document_name
    )
    print(
        "Chat ID:",
        chat_id
    )
    print(
        "Total chunks:",
        total_chunks
    )
    print("========================================")

    return total_chunks