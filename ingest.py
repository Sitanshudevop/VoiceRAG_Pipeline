"""
================================================================================
OFFLINE INGESTION PHASE: Vector Store Creation
================================================================================
Latency Budget & Architecture Rationale:
- Zero Framework Overhead: No LangChain/LlamaIndex. Pure bare-metal recursive splitting.
- Local Embeddings: 'all-MiniLM-L6-v2' (22M parameters, 384 dimensions).
  * Inference latency: ~8-15ms on CPU per batch.
  * Cost: $0.00 (Self-hosted local execution).
- Vector DB: FAISS IndexFlatL2 (In-memory exact L2 Euclidean distance).
  * Query Latency: < 0.5ms for small-to-medium corpora.
  * Memory footprint: < 10MB.
================================================================================
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from rank_bm25 import BM25Okapi

# Sample Knowledge Base Document (Tech specs & architecture documentation)
SAMPLE_DOCUMENT = """
Antigravity is a next-generation agentic AI coding platform engineered by DeepMind's Advanced Agentic Coding team.
It operates as an autonomous pair programmer capable of multi-file refactoring, debugging, terminal tool execution, and real-time planning.

Architecture Overview:
Antigravity utilizes an asymmetric agent loop combining Planning Mode and Execution Mode. In Planning Mode, the system decomposes complex user requirements into structured implementation plans, assessing architectural risk, backward compatibility, and latency impact before writing a single line of code.

Zero-Latency Voice RAG Pipeline:
The voice-enabled sub-system within Antigravity achieves sub-200ms round-trip latency through an ultra-streamlined pipeline:
1. Speech-to-Text (STT): Groq Cloud Whisper Large v3 transcribes voice packets asynchronously with time-to-first-token under 100ms.
2. In-Memory Vector Search: A local FAISS IndexFlatL2 index paired with all-MiniLM-L6-v2 embeddings delivers microsecond retrieval times (<0.5ms) on local CPU cores.
3. Fast Inference Generation: Groq LPU inference running Meta Llama 3 8B generates streaming tokens at over 800 tokens per second.
4. Edge Client: A minimalist vanilla Web Audio API frontend captures 16kHz uncompressed PCM audio chunks directly to an asynchronous FastAPI backend.

Security & Guardrail Specifications:
All transcribed user queries pass through an immediate in-memory regex guardrail filter. Any input under two words or matching flagged adversarial injection patterns is instantly short-circuited in under 0.2ms, protecting downstream LLM compute resources and maintaining strict latency budgets.
"""

class RecursiveCharacterTextSplitter:
    """
    Bare-metal recursive text splitter with overlap.
    Avoids LangChain dependency while preserving semantic paragraph and sentence boundaries.
    """
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 40, separators: list = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list:
        final_chunks = []
        # Find the best separator that exists in the text
        separator = self.separators[-1]
        for s in self.separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break

        splits = text.split(separator) if separator else list(text)
        good_splits = []
        
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                # Recursive split for large segments
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators[self.separators.index(separator) + 1:]
                )
                final_chunks.extend(sub_splitter.split_text(s))
                
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return [c.strip() for c in final_chunks if c.strip()]

    def _merge_splits(self, splits: list, separator: str) -> list:
        docs = []
        current_doc = []
        total_len = 0

        for s in splits:
            s_len = len(s)
            if total_len + s_len + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc_text = separator.join(current_doc)
                    if doc_text.strip():
                        docs.append(doc_text)
                    # Handle overlap
                    while current_doc and total_len > self.chunk_overlap:
                        popped = current_doc.pop(0)
                        total_len -= len(popped) + (len(separator) if current_doc else 0)
            current_doc.append(s)
            total_len += s_len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            doc_text = separator.join(current_doc)
            if doc_text.strip():
                docs.append(doc_text)
        return docs

def run_ingest(output_dir: str = "."):
    print("[Ingest] Initializing bare-metal text chunker...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    raw_chunks = splitter.split_text(SAMPLE_DOCUMENT)
    
    chunks_with_metadata = []
    for i, text in enumerate(raw_chunks):
        section_title = "General"
        if "Architecture" in text or "Mode" in text:
            section_title = "Architecture Overview"
        elif "Pipeline" in text or "STT" in text or "Generation" in text:
            section_title = "Zero-Latency Voice RAG Pipeline"
        elif "Security" in text or "guardrail" in text:
            section_title = "Security & Guardrail Specifications"
            
        chunks_with_metadata.append({
            "chunk_id": f"chunk-{i:03d}",
            "source": "Antigravity_TechSpecs.md",
            "section_title": section_title,
            "text": text
        })

    print(f"[Ingest] Generated {len(chunks_with_metadata)} high-quality semantic chunks.")

    print("[Ingest] Loading local embedding model 'all-MiniLM-L6-v2' (CPU)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print("[Ingest] Encoding chunks to float32 dense vectors...")
    texts_to_encode = [c["text"] for c in chunks_with_metadata]
    embeddings = model.encode(texts_to_encode, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]  # 384 dimensions for all-MiniLM-L6-v2
    print(f"[Ingest] Building FAISS IndexFlatIP (Normalized Inner Product / Cosine Similarity) - Dimension: {dimension}...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_file = os.path.join(output_dir, "vectorstore.index")
    metadata_file = os.path.join(output_dir, "metadata.json")

    print(f"[Ingest] Persisting FAISS index to {index_file}...")
    faiss.write_index(index, index_file)

    print(f"[Ingest] Persisting chunk metadata to {metadata_file}...")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, indent=2)

    print("[Ingest] Building BM25 index for hybrid search...")
    tokenized_corpus = [doc["text"].lower().split() for doc in chunks_with_metadata]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_file = os.path.join(output_dir, "bm25.pkl")
    print(f"[Ingest] Persisting BM25 index to {bm25_file}...")
    with open(bm25_file, "wb") as f:
        pickle.dump(bm25, f)

    print("[Ingest] Ingestion successfully completed! Vector store is ready for runtime.")

if __name__ == "__main__":
    run_ingest()
