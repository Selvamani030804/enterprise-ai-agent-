import io
import os
import requests
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="enterprise_docs")

def process_pdf(file_bytes: bytes, filename: str) -> int:
    """PDF-la irundhu text extract panni chunks-ah ChromaDB-la store pannum."""
    reader = PdfReader(io.BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    chunk_size = 500
    overlap = 50
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap

    if not chunks:
        return 0

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    return len(chunks)

def query_rag_with_llm(user_question: str) -> dict:
    """ChromaDB context retrieve panni, automatic model fallback vazhiya guaranteed response generate pannum."""
    # 1. Retrieve matching context chunks
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )
    
    docs = results.get("documents", [[]])[0]
    context = "\n---\n".join(docs) if docs else "No relevant document context found."

    prompt = f"""You are an enterprise AI assistant. Answer the user question based strictly on the provided document context. If the answer is not in the context, say "I cannot find this information in the uploaded documents."

Context:
{context}

Question:
{user_question}

Answer:"""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    # Fallback model list: Ondhu fail aanaal adutha model automatic-aa run aagum
    candidate_models = [
        "gemini-flash-latest",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.8-flash"
    ]

    ai_text = None
    last_error = ""

    for model_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            res = requests.post(url, json=payload, timeout=20)
            data = res.json()

            if "candidates" in data and len(data["candidates"]) > 0:
                ai_text = data["candidates"][0]["content"]["parts"][0]["text"]
                break
            elif "error" in data:
                last_error = data["error"].get("message", "Error")
                continue
        except Exception as e:
            last_error = str(e)
            continue

    if not ai_text:
        ai_text = f"API Error across all models: {last_error}"

    return {
        "answer": ai_text,
        "context_used": context
    }
