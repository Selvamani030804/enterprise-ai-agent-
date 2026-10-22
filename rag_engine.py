import os
import pypdf
import chromadb
from chromadb.utils import embedding_functions
import requests

CHROMA_DATA_PATH = "chroma_db"
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="enterprise_knowledge_base",
    embedding_function=default_ef
)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_pdf_and_index(pdf_path: str):
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    if not full_text.strip():
        raise Exception("No readable text found in the PDF.")

    chunks = chunk_text(full_text)
    doc_ids = [f"{os.path.basename(pdf_path)}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(pdf_path)} for _ in chunks]

    collection.upsert(
        documents=chunks,
        ids=doc_ids,
        metadatas=metadatas
    )
    return len(chunks)

def call_gemini_api(prompt: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("GEMINI_API_KEY is not set in environment.")

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return {"answer": text, "model_used": model_name}
        except Exception:
            continue

    raise Exception("All fallback Gemini models failed or rate limits exceeded.")

def query_rag_engine(user_query: str):
    results = collection.query(
        query_texts=[user_query],
        n_results=3
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        context = "No relevant context found in documents."
        sources = []
    else:
        context = "\n---\n".join(documents)
        sources = list(set([m.get("source", "Unknown") for m in metadatas if m]))

    prompt = f"""You are an enterprise AI assistant developed by R. Selvamani.
Answer the user's question accurately using ONLY the provided document context.
If the information is not in the context, clearly state that the document does not contain that info.

Context:
{context}

Question:
{user_query}

Answer:"""

    response = call_gemini_api(prompt)
    return {
        "answer": response["answer"],
        "model_used": response["model_used"],
        "sources": sources
    }
