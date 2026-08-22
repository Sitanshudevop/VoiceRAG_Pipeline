"""
================================================================================
RUNTIME PHASE: Ultra-Low Latency Voice-Enabled RAG Pipeline
================================================================================
"""

import os
import json
import time
import asyncio
import logging
import re
import hashlib
import pickle
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from collections import OrderedDict

import numpy as np
import faiss
from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from dotenv import load_dotenv
from groq import AsyncGroq

# Load environment variables from .env
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VoiceRAG")

# ==============================================================================
# GLOBAL RUNTIME STATE
# ==============================================================================
embedding_model: Optional[SentenceTransformer] = None
vector_index: Optional[faiss.Index] = None
chunks_metadata: List[dict] = []
bm25_index = None
groq_client: Optional[AsyncGroq] = None

app_metrics = {
    "total_queries": 0,
    "latencies": []
}

# Cache & Rate Limiting & Sessions
query_cache = OrderedDict()
CACHE_MAX_SIZE = 1000

rate_limits = {}
RATE_LIMIT_MAX = 20  # per minute

user_sessions = {}
SESSION_MAX_TURNS = 3

# Guardrail & Safety Constants
MIN_QUERY_WORDS = 2
HARMFUL_PATTERNS = {
    "hack", "exploit", "bypass", "ddos", "malware", 
    "kill", "suicide", "bomb", "destroy", "steal"
}
PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"you are now",
    r"jailbreak",
    r"forget everything"
]

# Relevance Threshold (Cosine Similarity / Inner Product)
RELEVANCE_THRESHOLD = 0.3

# PII Redaction Regexes
EMAIL_REGEX = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
PHONE_REGEX = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
CC_REGEX = r'\b(?:\d[ -]*?){13,16}\b'

# ==============================================================================
# LIFESPAN & WARM-UP
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, vector_index, chunks_metadata, groq_client

    logger.info("Initializing Voice RAG Runtime Engine...")
    t_start = time.perf_counter()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("CRITICAL: GROQ_API_KEY is not set.")
    groq_client = AsyncGroq(api_key=api_key)

    logger.info("Skipping local embedding model. API Embeddings configured.")

    index_path = "vectorstore.index"
    metadata_path = "metadata.json"
    bm25_path = "bm25.pkl"

    if not (os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(bm25_path)):
        logger.info("Vector databases missing. Running auto-ingestion via HF API...")
        import ingest
        ingest.run_ingest()

    if os.path.exists(index_path) and os.path.exists(metadata_path) and os.path.exists(bm25_path):
        vector_index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_metadata = json.load(f)
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)
        logger.info(f"Loaded {len(chunks_metadata)} chunks into FAISS and BM25 databases.")
    else:
        logger.warning(f"FAISS index, metadata, or bm25.pkl not found. Please execute 'python ingest.py' first!")

    t_total = (time.perf_counter() - t_start) * 1000
    logger.info(f"Runtime startup & warm-up completed in {t_total:.2f}ms.")

    yield
    logger.info("Shutting down Voice RAG Runtime Engine.")

app = FastAPI(title="Production Voice RAG API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# ==============================================================================
# UTILS & GUARDRAILS
# ==============================================================================

def redact_pii(text: str) -> str:
    text = re.sub(EMAIL_REGEX, "[EMAIL REDACTED]", text)
    text = re.sub(PHONE_REGEX, "[PHONE REDACTED]", text)
    text = re.sub(CC_REGEX, "[CC REDACTED]", text)
    return text

def check_guardrails(text: str) -> Optional[str]:
    clean_text = text.strip()
    words = clean_text.split()
    
    if len(words) < MIN_QUERY_WORDS:
        return "Your speech was too brief or unclear. Please ask a complete question."

    lower_text = clean_text.lower()
    for harmful in HARMFUL_PATTERNS:
        if harmful in lower_text:
            return "Request rejected by safety guardrails: Harmful intent detected."
            
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, lower_text):
            return "Request rejected: Prompt injection detected."

    return None

def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in rate_limits:
        rate_limits[client_ip] = []
    
    # Clean up old timestamps (older than 60s)
    rate_limits[client_ip] = [ts for ts in rate_limits[client_ip] if now - ts < 60]
    
    if len(rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        return False
        
    rate_limits[client_ip].append(now)
    return True

def lexical_rerank(query: str, chunks: List[dict]) -> List[dict]:
    """Basic keyword overlap reranking over top-k FAISS chunks"""
    query_words = set(re.findall(r'\w+', query.lower()))
    
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk["text"].lower()))
        overlap = len(query_words.intersection(chunk_words))
        scored_chunks.append((overlap, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_chunks[:2]] # Return top 2

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    groq_status = "degraded"
    if groq_client:
        try:
            await groq_client.models.list()
            groq_status = "ok"
        except:
            pass
            
    return {
        "status": "ready" if vector_index is not None and bm25_index is not None and groq_status == "ok" else "degraded",
        "vector_chunks_indexed": len(chunks_metadata) if chunks_metadata else 0,
        "embedding_model": "all-MiniLM-L6-v2 (HF API)",
        "llm_engine": "Groq LPU",
        "stt_engine": "Groq LPU (whisper-large-v3)",
        "groq_api_status": groq_status,
        "rate_limiting": "enabled",
        "pii_redaction": "enabled"
    }

@app.get("/metrics")
async def get_metrics():
    avg_lat = sum(app_metrics["latencies"]) / len(app_metrics["latencies"]) if app_metrics["latencies"] else 0
    return {
        "total_queries": app_metrics["total_queries"],
        "rolling_average_latency_ms": round(avg_lat, 2)
    }

from pydantic import BaseModel
class FollowupRequest(BaseModel):
    text: str

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    t_stt_start = time.perf_counter()
    try:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio.")
        if len(audio_bytes) > 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file exceeds 1MB limit.")
            
        transcription = await groq_client.audio.transcriptions.create(
            file=(audio.filename or "speech.webm", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
            language="en",
            temperature=0.0
        )
        transcribed_text = transcription.text.strip()
        transcribed_text = redact_pii(transcribed_text)
        
        latency = round((time.perf_counter() - t_stt_start) * 1000, 2)
        return {"transcription": transcribed_text, "stt_ms": latency}
    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/suggest_followups")
async def suggest_followups(req: FollowupRequest):
    try:
        response = await groq_client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": "You are a JSON API. Generate exactly 2 short follow-up questions the user could ask based on the provided text. Return ONLY a JSON object with a 'questions' array of 2 strings."},
                {"role": "user", "content": req.text}
            ],
            response_format={"type": "json_object"},
            max_tokens=150,
            temperature=0.5,
            reasoning_effort="none",
            reasoning_format="hidden"
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Suggest Followups Error: {e}")
        return {"questions": []}

@app.post("/ask")
async def ask_pipeline(
    request: Request,
    audio: UploadFile = File(None, description="Binary audio stream"),
    text_fallback: str = Form(None, description="Text query if no audio")
):
    t_start_total = time.perf_counter()
    latency_breakdown = {}

    client_ip = request.client.host if request.client else "127.0.0.1"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 20 requests per minute.")

    # STT or Text Fallback
    t_stt_start = time.perf_counter()
    transcribed_text = ""
    
    if text_fallback:
        transcribed_text = text_fallback.strip()
        if len(transcribed_text) > 2000:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length of 2000 characters.")
    elif audio:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio.")
        if len(audio_bytes) > 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio file exceeds 1MB (approx 30s) limit.")
        try:
            transcription = await groq_client.audio.transcriptions.create(
                file=(audio.filename or "speech.webm", audio_bytes),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )
            transcribed_text = transcription.text.strip()
        except Exception as e:
            logger.error(f"STT Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Must provide audio or text_fallback")
        
    latency_breakdown["stt_ms"] = round((time.perf_counter() - t_stt_start) * 1000, 2)

    # Redact PII
    transcribed_text = redact_pii(transcribed_text)

    # Guardrails
    t_guard_start = time.perf_counter()
    guardrail_violation = check_guardrails(transcribed_text)
    latency_breakdown["guardrail_ms"] = round((time.perf_counter() - t_guard_start) * 1000, 2)

    if guardrail_violation:
        async def event_generator_blocked():
            payload = {
                "transcription": transcribed_text,
                "token": guardrail_violation,
                "is_final": True,
                "latency": latency_breakdown
            }
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingResponse(event_generator_blocked(), media_type="text/event-stream")

    # Cache Check
    query_hash = hashlib.md5(transcribed_text.lower().encode()).hexdigest()
    if query_hash in query_cache:
        cached_result = query_cache.pop(query_hash)
        query_cache[query_hash] = cached_result
        
        async def event_generator_cached():
            payload = {
                "transcription": transcribed_text,
                "token": cached_result["answer"],
                "is_final": True,
                "cached": True,
                "latency": latency_breakdown,
                "context": cached_result["context"]
            }
            yield f"data: {json.dumps(payload)}\n\n"
        return StreamingResponse(event_generator_cached(), media_type="text/event-stream")

    # Embedding via API
    t_embed_start = time.perf_counter()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set for API embeddings")
        
    async def fetch_embedding(text: str):
        import requests
        API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        headers = {"Authorization": f"Bearer {hf_token}"}
        response = await asyncio.to_thread(
            requests.post, API_URL, headers=headers, json={"inputs": [text]}
        )
        if response.status_code != 200:
            raise Exception(f"HF API Error: {response.text}")
        import numpy as np
        emb = np.array(response.json(), dtype="float32")
        faiss.normalize_L2(emb)
        return emb

    try:
        query_vector = await fetch_embedding(transcribed_text)
    except Exception as e:
        logger.error(f"Embedding API failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding API failed")
        
    latency_breakdown["embedding_ms"] = round((time.perf_counter() - t_embed_start) * 1000, 2)

    # Retrieval & Thresholding
    t_retrieval_start = time.perf_counter()
    distances, indices = vector_index.search(query_vector, 5)
    
    retrieved_chunks = []
    
    if bm25_index is not None and len(chunks_metadata) > 0:
        query_tokens = transcribed_text.lower().split()
        bm25_scores = bm25_index.get_scores(query_tokens)
        
        # Max-Min normalization for BM25 and FAISS
        max_faiss = float(max(distances[0])) if len(distances[0]) > 0 else 1.0
        min_faiss = float(min(distances[0])) if len(distances[0]) > 0 else 0.0
        range_faiss = max_faiss - min_faiss if max_faiss > min_faiss else 1.0
        
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 else 1.0
        min_bm25 = float(min(bm25_scores)) if len(bm25_scores) > 0 else 0.0
        range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        
        hybrid_results = []
        for i, doc in enumerate(chunks_metadata):
            faiss_score = 0.0
            if i in indices[0]:
                faiss_score = float(distances[0][list(indices[0]).index(i)])
            
            norm_faiss = (faiss_score - min_faiss) / range_faiss
            norm_bm25 = (bm25_scores[i] - min_bm25) / range_bm25
            
            # Confidence out of 100
            confidence = int((norm_faiss * 0.6 + norm_bm25 * 0.4) * 100)
            hybrid_results.append((confidence, doc))
            
        hybrid_results.sort(key=lambda x: x[0], reverse=True)
        top_hybrid = hybrid_results[:2]
        
        if top_hybrid and top_hybrid[0][0] >= 30: # 30% confidence threshold
            for score, doc in top_hybrid:
                doc_copy = dict(doc)
                doc_copy["confidence"] = score
                retrieved_chunks.append(doc_copy)
        
    latency_breakdown["retrieval_ms"] = round((time.perf_counter() - t_retrieval_start) * 1000, 2)

    # Session Memory
    if client_ip not in user_sessions:
        user_sessions[client_ip] = []
    session = user_sessions[client_ip]
    
    if not retrieved_chunks:
        # Fallback for conversational or ungrounded queries
        system_instruction = (
            "You are a friendly voice assistant. The user asked a general question. "
            "Answer directly, conversationally, and in under 2 sentences."
        )
        messages = [{"role": "system", "content": system_instruction}]
        for msg in session:
            messages.append(msg)
        messages.append({"role": "user", "content": transcribed_text})
    else:
        # Build Context
        context_str = "\n---\n".join([f"Source: {c['source']} | Section: {c['section_title']}\n{c['text']}" for c in retrieved_chunks])
        
        system_instruction = (
            "You are a fast, helpful voice assistant. Use the provided retrieved context to answer domain-specific questions accurately. "
            "If a question is general knowledge or conversational and not covered in the context, answer directly and concisely using your general knowledge in 1-2 sentences. "
            "Never mention 'based on the provided documents' unless explicitly asked. "
            "Keep responses direct, natural, and under 2 sentences for fast voice output."
        )
        
        messages = [{"role": "system", "content": system_instruction}]
        for msg in session:
            messages.append(msg)
            
        messages.append({"role": "user", "content": f"Context:\n{context_str}\n\nUser Question:\n{transcribed_text}"})

    async def event_generator():
        # Yield transcription and context immediately before LLM buffering
        initial_payload = {
            "transcription": transcribed_text,
            "token": "",
            "is_final": False,
            "context": retrieved_chunks
        }
        yield f"data: {json.dumps(initial_payload)}\n\n"
        
        t_llm_start = time.perf_counter()
        try:
            stream = await groq_client.chat.completions.create(
                messages=messages,
                model="qwen/qwen3.6-27b",
                temperature=0.1,
                max_tokens=150,
                reasoning_effort="none",
                reasoning_format="hidden",
                stream=True
            )
            
            full_answer = ""
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    if "<think>" not in token and "</think>" not in token:
                        full_answer += token
                        payload = {
                            "token": token,
                            "is_final": False
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                        
            latency_breakdown["llm_ms"] = round((time.perf_counter() - t_llm_start) * 1000, 2)
            total_time = round((time.perf_counter() - t_start_total) * 1000, 2)
            latency_breakdown["total_ms"] = total_time
            
            app_metrics["total_queries"] += 1
            app_metrics["latencies"].append(total_time)
            if len(app_metrics["latencies"]) > 100:
                app_metrics["latencies"].pop(0)
            
            # Cache the result
            if len(query_cache) >= CACHE_MAX_SIZE:
                query_cache.popitem(last=False)
            query_cache[query_hash] = {
                "answer": full_answer,
                "context": retrieved_chunks
            }
            
            # Update Session
            session.append({"role": "user", "content": transcribed_text})
            session.append({"role": "assistant", "content": full_answer})
            if len(session) > SESSION_MAX_TURNS * 2:
                session.pop(0)
                session.pop(0)
                
            final_payload = {
                "token": "",
                "is_final": True,
                "latency": latency_breakdown
            }
            yield f"data: {json.dumps(final_payload)}\n\n"
            
        except Exception as e:
            logger.error(f"Groq stream error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'is_final': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
