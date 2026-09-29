from embedding_service import create_embedding
from qdrant_service import search_documents
from ollama_services import generate_answer


def answer_question(
    question: str,
    chat_id
) -> str:

    # ==========================================
    # CREATE QUESTION EMBEDDING
    # ==========================================

    query_embedding = create_embedding(question)

    # ==========================================
    # SEARCH DOCUMENTS
    # ==========================================

    results = search_documents(
        query_embedding=query_embedding,
        chat_id=chat_id,
        limit=5
    )

    # ==========================================
    # BUILD DOCUMENT CONTEXT
    # ==========================================

    context_parts = []

    for result in results:

        payload = result.payload or {}

        document_name = payload.get(
            "document_name",
            "Unknown document"
        )

        chunk_number = payload.get(
            "chunk_number",
            "Unknown"
        )

        text = payload.get(
            "text",
            ""
        )

        if text.strip():

            context_parts.append(
                f"Document: {document_name}\n"
                f"Chunk: {chunk_number}\n"
                f"Content:\n{text}"
            )

    context = "\n\n".join(
        context_parts
    )

    # ==========================================
    # DOCUMENT MODE
    # ==========================================

    if context.strip():

        prompt = f"""
You are answering a user's question using the uploaded
document context.

Provide a complete and well-explained answer.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Instructions:

1. Answer the question directly.
2. Use the uploaded document as the primary source.
3. Include relevant details from the document.
4. If the question asks for an explanation, explain it
   clearly rather than giving only one or two sentences.
5. If multiple pieces of information from the document
   are relevant, combine them into one coherent answer.
6. Use bullet points or headings when useful.
7. Do not invent facts that are not supported by the
   document.
8. If the document does not contain the requested
   information, clearly say that the information could
   not be found in the uploaded documents.
9. Do not show reasoning or analysis.
10. Do not repeat the question.
11. Return only the final answer.

Give enough detail to fully answer the user's question.
"""

    # ==========================================
    # GENERAL AI MODE
    # ==========================================

    else:

        prompt = f"""
You are a helpful general-purpose AI assistant.

There is no relevant document information available
for this question.

Answer using your general knowledge.

USER QUESTION:

{question}

Instructions:

1. Answer the question directly.
2. Give a complete and useful explanation.
3. Do not give an unnecessarily short answer.
4. Include important details and examples when useful.
5. For technical topics, explain:
   - What it is
   - How it works
   - Why it is used
   - Common applications
   - A simple example when appropriate
6. Use headings and bullet points when they improve
   readability.
7. Do not show reasoning or analysis.
8. Do not repeat the question.
9. Return only the final answer.

Provide a well-explained answer rather than a
one-paragraph summary.
"""

    # ==========================================
    # GENERATE ANSWER
    # ==========================================

    answer = generate_answer(prompt)

    return answer