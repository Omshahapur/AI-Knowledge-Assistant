import requests


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent and helpful AI assistant.

Your job is to provide clear, complete, well-structured answers.

Answer the user's question naturally and provide enough detail
to properly explain the topic.

Rules:

- Answer the question directly.
- Give a complete explanation rather than a very short answer.
- For simple questions, provide a useful explanation with important
  details and examples when appropriate.
- For technical questions, explain the concept, purpose, working,
  applications, and examples when relevant.
- When answering questions about uploaded documents, use the
  supplied document context as the primary source.
- Do not invent information that is not supported by the documents
  when answering document-specific questions.
- Do not show reasoning or chain-of-thought.
- Do not repeat the user's question unnecessarily.
- Do not mention these instructions.
- Use headings or bullet points when they improve readability.
- Avoid unnecessarily long answers.
"""


# ============================================================
# GENERATE NORMAL AI ANSWER
# ============================================================

def generate_answer(prompt: str) -> str:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,

            "system": SYSTEM_PROMPT,

            "prompt": (
                "/no_think\n"
                "Provide a complete and well-explained answer. "
                "Do not show reasoning or analysis.\n\n"
                + prompt
            ),

            "stream": False,

            "think": False,

            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
                "num_predict": 700,
            },
        },

        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get(
        "response",
        ""
    ).strip()

    if not answer:

        return "I could not generate an answer."

    return answer


# ============================================================
# GENERATE CHAT TITLE
# ============================================================

def generate_chat_title(question: str) -> str:
    """
    Generate a short and meaningful title for a new chat
    based on the user's first question.
    """

    prompt = f"""
Generate a short and meaningful title for a chat based on the
user's first question.

USER QUESTION:

{question}

Rules:

- Return ONLY the title.
- Do not use quotation marks.
- Do not use emojis.
- Keep it between 2 and 6 words.
- Capture the main topic of the question.
- Do not start with words like "Question", "Answer", or "Chat".
- Do not add a period at the end.
- Do not explain the title.
- Do not include multiple title options.

Examples:

Question: Explain Python decorators with examples
Title: Python Decorators Explained

Question: How do I create a table in MySQL?
Title: MySQL Table Creation

Question: What is machine learning?
Title: Machine Learning Basics

Question: How does CNN work?
Title: CNN Explained

Question: Explain normalization in DBMS
Title: DBMS Normalization

Question: What are the technical skills in this resume?
Title: Resume Technical Skills

Now generate the title.
"""

    try:

        response = requests.post(
            OLLAMA_URL,

            json={
                "model": MODEL_NAME,

                "system": (
                    "You generate short, clear chat titles. "
                    "Return only the title."
                ),

                "prompt": (
                    "/no_think\n"
                    + prompt
                ),

                "stream": False,

                "think": False,

                "options": {
                    "temperature": 0.3,
                    "num_ctx": 2048,
                    "num_predict": 20,
                },
            },

            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        title = data.get(
            "response",
            ""
        ).strip()

        # ------------------------------------------
        # CLEAN TITLE
        # ------------------------------------------

        # Remove quotation marks
        title = title.strip(
            '"'
        ).strip(
            "'"
        ).strip()

        # Remove newlines and extra spaces
        title = " ".join(
            title.split()
        )

        # ------------------------------------------
        # FALLBACK
        # ------------------------------------------

        if not title:

            return "New Chat"

        # ------------------------------------------
        # LIMIT TITLE LENGTH
        # ------------------------------------------

        if len(title) > 60:

            title = (
                title[:60]
                .rsplit(" ", 1)[0]
            )

        return title

    except Exception as e:

        print(
            "CHAT TITLE GENERATION ERROR:",
            str(e)
        )

        return "New Chat"