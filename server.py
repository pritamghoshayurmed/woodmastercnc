from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import load_settings
from src.pipeline.rag_pipepline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag: RAGPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    load_dotenv()
    settings = load_settings()
    rag = RAGPipeline(settings)
    
    # Run initialize. We might want to handle this non-blocking if it was huge, 
    # but load should be fast if FAISS is built.
    logger.info("Initializing RAG Pipeline...")
    rag.initialize(force_rebuild=False)
    logger.info("RAG Pipeline initialized.")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Woodmaster CNC Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# For serving local files like images, we can mount the data folder if we wish
import os
import pathlib
if not os.path.exists("data"):
    os.makedirs("data")

app.mount("/data", StaticFiles(directory="data"), name="data")

class WebhookRequest(BaseModel):
    session_id: str
    message: str

@app.post("/webhook")
async def handle_webhook(req: WebhookRequest):
    if not rag:
        return JSONResponse(status_code=503, content={"error": "RAG pipeline not initialized"})
    
    try:
        response = rag.query(question=req.message, session_id=req.session_id)
        
        # response should contain 'answer', 'sources', 'images'
        return {
            "session_id": req.session_id,
            "reply": response.get("answer", ""),
            "images": response.get("images", []),
            "metadata": response.get("retrieval", [])
        }
    except Exception as e:
        logger.error(f"Error handling query: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})


# Serve the simple static index.html on root
@app.get("/")
async def get_index():
    if os.path.exists("frontend.html"):
        with open("frontend.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
