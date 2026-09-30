# AI Knowledge Assistant

AI Knowledge Assistant is a web-based AI project that I built to answer questions using an LLM and information from uploaded PDF documents.

The project uses Django for the backend, Ollama for running the AI models locally, and Qdrant for storing document embeddings and performing semantic search.

## What this project does

- Ask general questions and get answers from the AI model
- Upload PDF documents and ask questions about them
- Store uploaded documents separately for each chat
- Search relevant information from PDFs using semantic search
- Maintain previous messages in a conversation
- Understand follow-up questions using previous conversation context
- Generate a title for each new chat automatically

## Technologies Used

- Python
- Django
- HTML
- CSS
- JavaScript
- Ollama
- Qwen2.5:3b
- nomic-embed-text
- Qdrant
- PyMuPDF
- SQLite

## How the PDF Question Answering Works

When a PDF is uploaded, the application first extracts the text from the PDF using PyMuPDF.

The extracted text is then divided into smaller chunks. Each chunk is converted into an embedding using the `nomic-embed-text` model and stored in Qdrant.

When a user asks a question, the question is also converted into an embedding. Qdrant searches for the most relevant document chunks using semantic similarity.

The retrieved content is then given to the Qwen2.5 model along with the user's question, and the model generates the final answer.

The basic flow is:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
Qdrant
 ↓
Semantic Search
 ↓
Relevant Context
 ↓
Qwen2.5
 ↓
Answer
