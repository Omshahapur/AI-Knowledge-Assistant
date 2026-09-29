import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"


SYSTEM_PROMPT = """
You are an intelligent and helpful AI assistant.

Your job is to provide clear, complete, well-structured answers.

Answer the user's question naturally and provide enough detail
to properly explain the topic.

Rules:

- Answer the question directly.
- Give a complete explanation rather than a very short response.
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

                # More detailed answers
                "temperature": 0.2,

                # Larger context window
                "num_ctx": 8192,

                # Allow longer answers
                "num_predict": 700
            }
        },

        timeout=300
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get(
        "response",
        ""
    ).strip()

    if not answer:

        return (
            "I could not generate an answer."
        )

    return answer