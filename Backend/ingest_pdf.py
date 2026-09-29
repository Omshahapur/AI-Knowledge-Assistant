
from pdf_processor import extract_text_from_pdf
from text_chunker import chunk_text
from embedding_service import create_embedding
from qdrant_service import add_document_chunk


# ==========================================
# PDF FILE PATH
# ==========================================

PDF_PATH = r"media\documents\OM_SHAHAPUR.pdf"


# ==========================================
# 1. EXTRACT TEXT FROM PDF
# ==========================================

print("Starting PDF ingestion...")

text = extract_text_from_pdf(PDF_PATH)

print("PDF text extracted successfully")

if not text.strip():

    raise ValueError(
        "No text was extracted from the PDF."
    )


# ==========================================
# 2. SPLIT TEXT INTO OVERLAPPING CHUNKS
# ==========================================

chunks = chunk_text(
    text,
    chunk_size=300,
    overlap=50
)

print("Total chunks:", len(chunks))

if not chunks:

    raise ValueError(
        "No chunks were created from the PDF."
    )


# ==========================================
# 3. GET DOCUMENT NAME
# ==========================================

document_name = PDF_PATH.split("\\")[-1]

print("Document:", document_name)


# ==========================================
# 4. CREATE EMBEDDINGS AND STORE IN QDRANT
# ==========================================

total_chunks = len(chunks)

for i, chunk in enumerate(chunks):

    print(
        f"\nProcessing chunk {i + 1}/{total_chunks}..."
    )

    # Create a 768-dimensional embedding
    embedding = create_embedding(chunk)

    # Store the chunk and metadata in Qdrant
    add_document_chunk(
        text=chunk,
        embedding=embedding,
        document_name=document_name,
        chunk_number=i + 1,
        total_chunks=total_chunks
    )

    print(
        f"Chunk {i + 1}/{total_chunks} stored successfully"
    )


# ==========================================
# 5. COMPLETION MESSAGE
# ==========================================

print("\n========================================")
print("PDF ingestion completed successfully")
print("Document:", document_name)
print("Total chunks stored:", total_chunks)
print("========================================")