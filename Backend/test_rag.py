
from rag_service import answer_question
from qdrant_service import client


question = "What technologies were used in the project?"

answer = answer_question(question)


print("\nAI Answer:\n")
print(answer)


client.close()

print("\nQdrant client closed successfully")