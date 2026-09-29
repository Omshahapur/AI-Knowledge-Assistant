
from embedding_service import create_embedding
from qdrant_service import search_documents, close_client


# ==========================================
# USER QUESTION
# ==========================================

question = "What is the educational qualification of Om Shahapur?"


try:

    # ==========================================
    # CREATE QUESTION EMBEDDING
    # ==========================================

    query_embedding = create_embedding(question)


    # ==========================================
    # SEARCH RELEVANT DOCUMENT CHUNKS
    # ==========================================

    results = search_documents(

        query_embedding,

        limit=5

    )


    # ==========================================
    # DISPLAY SEARCH RESULTS
    # ==========================================

    print("\nSearch Results:\n")


    if not results:

        print("No relevant results found.")


    else:

        for i, result in enumerate(results):

            payload = result.payload or {}

            print(f"--- Result {i + 1} ---")

            print(
                "Score:",
                result.score
            )

            print(
                "Document:",
                payload.get(
                    "document_name",
                    "Unknown"
                )
            )

            print(
                "Chunk:",
                payload.get(
                    "chunk_number",
                    "Unknown"
                )
            )

            print(
                "Text:",
                payload.get(
                    "text",
                    "No text available"
                    )
                )

            print()


finally:

    # ==========================================
    # CLOSE QDRANT CLIENT
    # ==========================================

    close_client()