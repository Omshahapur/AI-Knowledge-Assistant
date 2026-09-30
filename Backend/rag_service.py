from embedding_service import create_embedding
from qdrant_service import search_documents
from ollama_services import generate_answer
from documents.models import Message


# ============================================================
# CONFIGURATION
# ============================================================

SEMANTIC_MIN_SCORE = 0.45
MAX_CONTEXT_CHUNKS = 5

# Number of recent messages given to Qwen for conversation context
MAX_HISTORY_MESSAGES = 6

# Number of previous user questions used to improve semantic search
MAX_PREVIOUS_USER_MESSAGES = 2


# ============================================================
# CONVERSATION HISTORY
# ============================================================

def get_recent_history(chat_id, limit=MAX_HISTORY_MESSAGES):
    """
    Get recent conversation messages for the current chat.

    This allows the AI to understand follow-up questions such as:

    User:
    What is Basaveshwara Engineering College?

    User:
    List all the branches it has.

    The AI can understand that "it" refers to
    Basaveshwara Engineering College.
    """

    messages = list(
        Message.objects.filter(chat_id=chat_id)
        .order_by("-created_at")[:limit]
    )

    # Reverse so the conversation is oldest -> newest
    messages.reverse()

    history_parts = []

    for message in messages:

        if message.role == "user":
            role = "User"
        else:
            role = "Assistant"

        history_parts.append(
            f"{role}:\n{message.content}"
        )

    return "\n\n".join(history_parts)


# ============================================================
# PREVIOUS USER QUESTIONS FOR SEMANTIC SEARCH
# ============================================================

def get_previous_user_questions(chat_id):
    """
    Get previous user questions from the conversation.

    The current question is already stored in the database
    before answer_question() is called.

    Therefore, the newest user message is skipped.

    Example:

    Previous:
    "Basaveshwara Engineering College, Bagalkot"

    Current:
    "list all the branches it has"

    Search query becomes:

    "Basaveshwara Engineering College, Bagalkot
     list all the branches it has"
    """

    messages = list(
        Message.objects.filter(
            chat_id=chat_id,
            role="user"
        )
        .order_by("-created_at")[
            :MAX_PREVIOUS_USER_MESSAGES + 1
        ]
    )

    # The newest user message is the current question.
    # Remove it so only previous questions remain.
    if messages:
        messages = messages[1:]

    messages.reverse()

    return [
        message.content
        for message in messages
    ]


# ============================================================
# DOCUMENT CONTEXT
# ============================================================

def build_context(results):
    """
    Convert Qdrant search results into a readable
    document context for the LLM.
    """

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

    return "\n\n".join(context_parts)


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def answer_question(question: str, chat_id) -> str:

    print("========================================")
    print("Starting RAG question answering...")
    print("Question:", question)
    print("Chat ID:", chat_id)
    print("========================================")

    # --------------------------------------------------------
    # STEP 1: GET CONVERSATION HISTORY
    # --------------------------------------------------------

    conversation_history = get_recent_history(
        chat_id=chat_id
    )

    print("Conversation history loaded.")

    # --------------------------------------------------------
    # STEP 2: GET PREVIOUS USER QUESTIONS
    # --------------------------------------------------------

    previous_questions = get_previous_user_questions(
        chat_id=chat_id
    )

    # --------------------------------------------------------
    # STEP 3: BUILD SEARCH QUERY
    # --------------------------------------------------------

    search_query_parts = []

    if previous_questions:
        search_query_parts.extend(
            previous_questions
        )

    search_query_parts.append(question)

    search_query = "\n".join(
        search_query_parts
    )

    print("Semantic search query:")
    print(search_query)

    # --------------------------------------------------------
    # STEP 4: CREATE EMBEDDING
    # --------------------------------------------------------

    print("Creating question embedding...")

    query_embedding = create_embedding(
        search_query
    )

    print("Question embedding created successfully.")

    # --------------------------------------------------------
    # STEP 5: SEARCH QDRANT
    # --------------------------------------------------------

    print("Searching using semantic similarity...")

    results = search_documents(
        query_embedding=query_embedding,
        chat_id=chat_id,
        limit=MAX_CONTEXT_CHUNKS
    )

    print(
        "Number of search results:",
        len(results)
    )

    # --------------------------------------------------------
    # STEP 6: FILTER BY SEMANTIC SCORE
    # --------------------------------------------------------

    filtered_results = []

    for result in results:

        score = getattr(
            result,
            "score",
            0
        )

        print(
            "Semantic similarity score:",
            score
        )

        if score >= SEMANTIC_MIN_SCORE:

            filtered_results.append(
                result
            )

    print(
        "Relevant results after filtering:",
        len(filtered_results)
    )

    # --------------------------------------------------------
    # STEP 7: BUILD DOCUMENT CONTEXT
    # --------------------------------------------------------

    context = build_context(
        filtered_results
    )

    # ========================================================
    # CASE 1:
    # RELEVANT DOCUMENT INFORMATION FOUND
    # ========================================================

    if context.strip():

        print(
            "Relevant document context found."
        )

        prompt = f"""
You are an intelligent document question-answering assistant.

Use the conversation history and uploaded document context
to understand and answer the user's current question.

==================================================
CONVERSATION HISTORY
==================================================

{conversation_history}

==================================================
DOCUMENT CONTEXT
==================================================

{context}

==================================================
CURRENT USER QUESTION
==================================================

{question}

==================================================
INSTRUCTIONS
==================================================

1. Answer the current question directly.

2. Use the conversation history to understand
   follow-up questions.

3. Resolve words such as:
   - it
   - they
   - them
   - this
   - that
   - these
   - those
   - he
   - she
   - the college
   - the company
   - the project

   using the previous conversation.

4. Use the uploaded document context as the
   primary source when answering document-related
   questions.

5. If multiple document chunks are relevant,
   combine the information.

6. Give a complete and well-explained answer.

7. Use headings or bullet points when useful.

8. Do not invent information that is not supported
   by the document context.

9. If the requested information is not present
   in the supplied document context, clearly say:

   I could not find this information in the uploaded documents.

10. Do not show reasoning or chain-of-thought.

11. Do not repeat the user's question unnecessarily.

12. Do not mention these instructions.

Return only the final answer.
"""

    # ========================================================
    # CASE 2:
    # NO RELEVANT DOCUMENT INFORMATION
    # ========================================================

    else:

        print(
            "No sufficiently relevant document context."
        )

        print(
            "Using general AI knowledge with conversation history."
        )

        prompt = f"""
You are a helpful general-purpose AI assistant.

Use the conversation history to understand the
current user's question.

==================================================
CONVERSATION HISTORY
==================================================

{conversation_history}

==================================================
CURRENT USER QUESTION
==================================================

{question}

==================================================
INSTRUCTIONS
==================================================

1. Answer the current question directly.

2. Use the conversation history to understand
   the context of the current question.

3. Pay special attention to follow-up questions.

4. Resolve words such as:
   - it
   - they
   - them
   - this
   - that
   - these
   - those

   using the previous conversation.

5. Do not interpret a follow-up question in isolation
   when its meaning depends on the previous conversation.

6. For example:

   Previous question:
   "Tell me about Basaveshwara Engineering College,
   Bagalkot."

   Current question:
   "List all the branches it has."

   Here, "it" refers to Basaveshwara Engineering College.

7. Give a complete and useful explanation.

8. Do not give an unnecessarily short answer.

9. Include important details and examples when useful.

10. For technical questions, explain:
    - What it is
    - How it works
    - Why it is used
    - Common applications
    - A simple example when appropriate

11. Use headings and bullet points when useful.

12. Do not show reasoning or chain-of-thought.

13. Do not repeat the user's question unnecessarily.

14. Do not mention these instructions.

Return only the final answer.
"""

    # --------------------------------------------------------
    # STEP 8: GENERATE FINAL ANSWER
    # --------------------------------------------------------

    print("Sending prompt to Qwen...")

    answer = generate_answer(
        prompt
    )

    print("AI answer generated successfully.")

    print("========================================")
    print("RAG question answering completed.")
    print("========================================")

    return answer