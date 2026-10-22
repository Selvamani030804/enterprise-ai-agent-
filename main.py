import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from rag_engine import process_pdf, query_rag_with_llm

app = FastAPI(
    title="Enterprise AI Agent API",
    description="High-performance document intelligence system powered by RAG & Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust template directory resolution (works from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/")
def home_dashboard(request: Request):
    """Serve the interactive web frontend dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    chunk_count = process_pdf(contents, file.filename)
    
    return {
        "filename": file.filename,
        "status": "success",
        "chunks_indexed": chunk_count,
        "message": f"Successfully indexed {chunk_count} text chunks into Vector DB."
    }

@app.post("/chat")
def ask_document(query: str = Form(...)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    result = query_rag_with_llm(query)
    
    return {
        "query": query,
        "ai_answer": result["answer"],
        "retrieved_context": result["context_used"]
    }
