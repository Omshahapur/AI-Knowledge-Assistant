import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:4b",
        "prompt": "Explain what a PDF document is in two sentences.",
        "stream": False
    }
)

print(response.json()["response"])