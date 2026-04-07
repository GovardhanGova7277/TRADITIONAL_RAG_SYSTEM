# Traditional RAG System — Technical Documentation

> **A complete Retrieval-Augmented Generation pipeline** that processes multi-format documents, builds a FAISS vector index, and answers natural-language queries using Llama 4 Scout (via Groq).

---

## Table of Contents

1. [RAG System Overview](#1-rag-system-overview)
2. [Document Processing Pipeline](#2-document-processing-pipeline)
3. [Chunking Strategy](#3-chunking-strategy)
4. [Embedding Model](#4-embedding-model)
5. [Vector Store](#5-vector-store)
6. [Retrieval & Generation Pipeline](#6-retrieval--generation-pipeline)
7. [RAG Architecture](#7-rag-architecture)
8. [System Limitations](#8-system-limitations)
9. [Possible Improvements](#9-possible-improvements)
10. [Future Enhancements](#10-future-enhancements)

---

## Implementation Stack

| Component | Technology |
|-----------|-----------|
| **Document Loader** | LangChain — `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, `CSVLoader`, `UnstructuredPowerPointLoader` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` — `chunk_size=1000`, `overlap=200` |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Vector Store** | FAISS `IndexFlatL2` — persisted as `faiss.index` + `metadata.pkl` |
| **LLM** | `meta-llama/llama-4-scout-17b-16e-instruct` via Groq API |
| **Runtime Stats** | 373 raw docs → 1,656 chunks → embedding matrix shape `(1656, 384)` |
| **Language** | Python 3 |

---

## 1. RAG System Overview

### What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architecture pattern that enhances an LLM by grounding its responses in an external knowledge base retrieved at query time. Rather than relying entirely on knowledge baked into model weights during training, the system first retrieves the most relevant document fragments and injects them into the LLM prompt as context.

```
User Query
    │
    ▼
[Retriever] ──► relevant chunks ──► [LLM] ──► Answer
    │
    └── searches Vector Store built from your documents
```

### Why Use RAG?

| Problem | How RAG Solves It |
|---------|------------------|
| LLM training cutoff | Knowledge base is updated independently of the model |
| Hallucination | Answers are grounded in retrieved real text |
| Private/proprietary data | Documents never leave your infrastructure |
| Fine-tuning cost | Re-indexing documents is vastly cheaper than retraining |

### Purpose of This Implementation

This pipeline was built to query a local corpus of technical study materials (Kubernetes, Linux, process control). It demonstrates a complete production-style RAG loop — from raw files on disk to a summarised answer — using entirely open-source tooling except for the Groq inference API.

---

## 2. Document Processing Pipeline

### 2.1 File Discovery

`data_loader.py` uses `pathlib.Path.rglob('*')` to walk the data directory recursively. Each file's extension is inspected and dispatched to the appropriate loader via a `LOADER_MAPPING` dictionary.

```python
LOADER_MAPPING = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".docx": Docx2txtLoader,
    ".csv":  CSVLoader
}
```

Unsupported formats (`.ppt`, `.sqlite3`, `.bin`, etc.) are skipped with a `[SKIP]` log line. The vector store directory under `data/` is also skipped.

### 2.2 Loader Behaviour

| Extension | Loader | Output per call |
|-----------|--------|----------------|
| `.pdf` | `PyPDFLoader` | One `Document` per page |
| `.txt` | `TextLoader` | One `Document` per file |
| `.docx` | `Docx2txtLoader` | One `Document` per file |
| `.pptx` | `UnstructuredPowerPointLoader` | One `Document` per slide block |
| `.csv` | `CSVLoader` | One `Document` per row |

Each `Document` object carries:
- `page_content` — extracted plain-text string
- `metadata` — dict with at minimum `{"source": "<file_path>"}`, plus `{"page": N}` for PDFs

> **Observed result:** 4 PDFs + 3 TXT files → **373 raw Document chunks** before splitting.

---

## 3. Chunking Strategy

### 3.1 Why Chunking Is Necessary

The `all-MiniLM-L6-v2` embedding model has a hard maximum input of **256 word-pieces (~200 words)**. Any input exceeding this is silently truncated. Chunking also enables fine-grained retrieval — returning the precise paragraph that answers a question rather than an entire 50-page PDF.

### 3.2 Configuration

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # maximum characters per chunk
    chunk_overlap=200,      # overlap between consecutive chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]   # tried in order
)
```

The splitter tries `"\n\n"` (paragraph breaks) first, then `"\n"` (line breaks), then `" "` (words), then hard character splitting. This preserves natural language boundaries wherever possible.

### 3.3 What Overlap Does

```
Chunk N:   [.........................................200 chars overlapping....]
Chunk N+1: [200 chars overlapping...........................................]
```

The 200-character overlap ensures that sentences or concepts straddling a boundary are fully present in at least one chunk, preventing information loss at split points.

### 3.4 Advantages and Limitations

**Advantages**
- Semantically aware — prefers paragraph and sentence boundaries
- Simple to tune — just two hyperparameters
- No external NLP models required

**Limitations**
- Character-based, not token-based — dense text (code, JSON) may hit the model's token limit even under 1,000 characters
- No semantic coherence guarantee per chunk
- Overlap increases vector index size

> **Observed result:** 373 raw documents → **1,656 final chunks** (≈4.4 sub-chunks per source document unit on average).

---

## 4. Embedding Model

### 4.1 Model: `all-MiniLM-L6-v2`

| Property | Value |
|----------|-------|
| Architecture | BERT (MiniLM — 6-layer distilled) |
| Output dimensions | **384** |
| Max input length | 256 word-pieces |
| Training objective | Contrastive similarity on NLI + STS-B pairs |
| Model size | ~22 MB |
| Observed throughput | ~6.31 batches/sec on CPU (52 batches for 1,656 chunks) |

### 4.2 What Embeddings Are

An embedding is a **fixed-length numerical vector** that encodes the semantic meaning of text. Sentences with similar meaning are mapped to nearby points in the 384-dimensional vector space, enabling semantic (meaning-based) search instead of keyword matching.

```
"How does kubelet work?"  ──►  [0.13, -0.42, 0.81, ...]  384 floats
"kubelet agent node pods" ──►  [0.14, -0.40, 0.79, ...]  384 floats
                                      ↑ small L2 distance → high similarity
```

### 4.3 How the Model Works

1. Tokenise input text into sub-word tokens
2. Pass tokens through 6 transformer layers → contextual token embeddings
3. Mean-pool all token embeddings → single vector
4. L2-normalise the result

### 4.4 Code Reference

```python
class EmbeddingPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200):
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks):
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings   # shape: (N, 384)
```

### 4.5 Advantages and Limitations

**Advantages**
- Fast on CPU — small enough to run without a GPU
- Good general-purpose performance — transfers well to technical text
- Fully offline after initial download

**Limitations**
- 256 token hard truncation — tail of long chunks is silently dropped
- English-centric — quality degrades on multilingual content
- 384 dims — smaller than state-of-the-art models (e.g. OpenAI `text-embedding-3-large` = 3,072 dims)

---

## 5. Vector Store

### 5.1 What a Vector Store Is

A vector store is a data structure optimised for **nearest-neighbour search** in high-dimensional space. Given a query vector, it returns the `k` stored vectors that are closest according to a distance metric.

### 5.2 FAISS — `IndexFlatL2`

| Aspect | Detail |
|--------|--------|
| Library | FAISS (Facebook AI Similarity Search) |
| Index type | `IndexFlatL2` — exact brute-force L2 search |
| Vectors stored | 1,656 × 384 float32 |
| Persistence | `faiss.write_index()` → `faiss.index`; metadata → `metadata.pkl` |
| Distance metric | L2 (Euclidean) — lower = more similar |

### 5.3 Build Flow

```python
def build_from_documents(self, documents):
    emb_pipe = EmbeddingPipeline(...)
    chunks = emb_pipe.chunk_documents(documents)      # 1,656 chunks
    embeddings = emb_pipe.embed_chunks(chunks)        # (1656, 384)
    self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
    self.save()    # writes faiss.index + metadata.pkl
```

### 5.4 Query Flow

```python
def query(self, query_text, top_k=5):
    query_emb = self.model.encode([query_text]).astype('float32')  # (1, 384)
    D, I = self.index.search(query_emb, top_k)                     # distances, indices
    return [{"index": idx, "distance": dist, "metadata": self.metadata[idx]}
            for idx, dist in zip(I[0], D[0])]
```

---

## 6. Retrieval & Generation Pipeline

### 6.1 End-to-End Flow

```
User query
     │
     ▼
[1] Encode query → 384-dim float32 vector
     │
     ▼
[2] FAISS IndexFlatL2.search(query_vec, top_k=5)
     │  returns 5 (index, L2_distance) pairs
     ▼
[3] Fetch metadata[index]["text"] for each result
     │
     ▼
[4] context = "\n\n".join(chunk_texts)
     │
     ▼
[5] Build prompt:
     "Summarize the following context for the query: '{query}'
      Context: {context}
      Summary:"
     │
     ▼
[6] ChatGroq(llama-4-scout-17b-16e-instruct).invoke([prompt])
     │
     ▼
Final answer
```

### 6.2 Key Design Decisions

**Same model for indexing and querying:** The identical `all-MiniLM-L6-v2` instance is used for both. Using different models would place documents and queries in incompatible vector spaces, making retrieval meaningless.

**Lazy index loading:**
```python
if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
    docs = load_all_documents("data")
    self.vectorstore.build_from_documents(docs)   # first run: build
else:
    self.vectorstore.load()                        # subsequent runs: reuse
```

---

## 7. RAG Architecture

```
╔══════════════════════════════╗     ╔═══════════════════════════════╗
║     INGESTION PIPELINE       ║     ║       QUERY PIPELINE          ║
╠══════════════════════════════╣     ╠═══════════════════════════════╣
║                              ║     ║                               ║
║  Raw Documents               ║     ║  User Query                   ║
║  (PDF, TXT, DOCX, CSV, PPTX) ║     ║                               ║
║           │                  ║     ║        │                       ║
║           ▼                  ║     ║        ▼                       ║
║  Document Loader             ║     ║  Query Embedder               ║
║  (LangChain loaders)         ║     ║  (all-MiniLM-L6-v2 → 384d)   ║
║           │                  ║     ║        │                       ║
║           ▼                  ║     ║        ▼                       ║
║  Text Chunker                ║     ║  Vector Search ◄─────────────────┐
║  (chunk=1000, overlap=200)   ║     ║  (FAISS top_k=5)              ║  │
║           │                  ║     ║        │                       ║  │
║           ▼                  ║     ║        ▼                       ║  │
║  Embedding Model             ║     ║  Context Assembly             ║  │
║  (all-MiniLM-L6-v2 → 384d)  ║     ║  (join top-5 chunks)          ║  │
║           │                  ║     ║        │                       ║  │
║           ▼                  ║     ║        ▼                       ║  │
║  FAISS Vector Store ─────────╬─────╬──────(retrieves)              ║  │
║  (IndexFlatL2 persisted)     ║     ║        │                       ║  │
╚══════════════════════════════╝     ║        ▼                       ║  │
                                     ║  LLM (Groq)                   ║──┘
                                     ║  (Llama 4 Scout 17B)          ║
                                     ║        │                       ║
                                     ║        ▼                       ║
                                     ║  Generated Answer             ║
                                     ╚═══════════════════════════════╝
```

### Module Mapping

| Module | Pipeline Stage | Key Class |
|--------|---------------|-----------|
| `data_loader.py` | Document ingestion | `load_all_documents()` |
| `embeddings.py` | Chunking + embedding | `EmbeddingPipeline` |
| `vector_store.py` | FAISS build/load/search | `FaissVectorStore` |
| `search.py` | End-to-end RAG query | `RAGSearch` |

---

## 8. System Limitations

### 8.1 Retrieval

- **O(n·d) brute-force search** — `IndexFlatL2` scans every vector. Fine at 1,656 vectors; degrades beyond ~1 million.
- **No re-ranking** — top-5 results are returned in raw L2 distance order without a cross-encoder.
- **Text-only index** — tables, figures, and diagrams in PDFs are not extracted or embedded.

### 8.2 Embedding

- **256-token hard truncation** — the tail of any chunk longer than ~200 words is silently dropped before encoding.
- **No asymmetric query/passage tuning** — `all-MiniLM-L6-v2` treats queries and passages identically; dedicated asymmetric models perform better.
- **Static index** — adding/removing documents requires a full rebuild.

### 8.3 Generation

- **Context window pressure** — 5 × 1,000-char chunks ≈ 5,000 chars of context, which may crowd the LLM on long prompts.
- **Hardcoded prompt template** — no prompt versioning or A/B testing.
- **No source citation** — the answer does not reference which chunk it was drawn from.

### 8.4 Scaling

- **In-memory index** — the full FAISS index is held in RAM; millions of documents would exhaust memory.
- **Synchronous, single-threaded** — cannot handle concurrent queries.
- **No observability** — no latency tracking, error monitoring, or query logging.

---

## 9. Possible Improvements

### 9.1 Better Embedding Models

| Model | Why Better |
|-------|-----------|
| `BAAI/bge-large-en-v1.5` | Stronger bi-encoder; outperforms MiniLM on BEIR benchmarks |
| `intfloat/e5-large-v2` | Instruction-tuned for query/passage asymmetry |
| `nomic-embed-text` | Up to 8,192 token context; eliminates truncation |
| `text-embedding-3-large` (OpenAI) | 3,072-dim state-of-the-art; best retrieval quality |

### 9.2 Scalable FAISS Indices

```python
# Replace IndexFlatL2 with approximate HNSW for large corpora
index = faiss.IndexHNSWFlat(dim, 32)   # 32 = graph connectivity
index.hnsw.efConstruction = 200
```

### 9.3 Hybrid Search (Dense + Sparse)

Combine semantic vector search with a BM25/TF-IDF keyword retriever. Merge results using **Reciprocal Rank Fusion (RRF)**:

```
final_score(doc) = Σ 1 / (k + rank_in_retriever_i)
```

This captures both semantic similarity and exact keyword matches (error codes, product names, version numbers).

### 9.4 Cross-Encoder Re-ranking

```
Initial retrieval: top-20 candidates (bi-encoder, fast)
        ↓
Re-ranking:  cross-encoder/ms-marco-MiniLM-L-6-v2 (slower, more accurate)
        ↓
Final context: top-5 after re-ranking
```

### 9.5 Query Optimisation

- **HyDE** — generate a hypothetical answer with the LLM first; embed that answer as the retrieval query to bridge the query–document vocabulary gap.
- **Multi-query** — generate 3–5 reformulations of the user's query; retrieve for each and union results before re-ranking.

---

## 10. Future Enhancements

### 10.1 Production Architecture

```
                        ┌────────────────────────┐
                        │   FastAPI REST API      │
                        │   POST /query           │
                        │   POST /ingest          │
                        └──────────┬─────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       Redis cache          RAG pipeline         Vector DB
       (hot queries)     (async workers)      (Qdrant / Chroma)
```

- Serve via **FastAPI** with async request handling and Pydantic validation.
- Replace FAISS + pickle with **Qdrant** or **ChromaDB** for concurrent writes, metadata filtering, and incremental updates.
- Add a `/ingest` endpoint that accepts file uploads and appends vectors to a live index without downtime.
- Containerise with **Docker + docker-compose**.

### 10.2 Performance

| Optimisation | Expected Gain |
|-------------|--------------|
| GPU batch embedding | 5–10× throughput |
| Async Groq client | Concurrent query handling |
| Streaming LLM output | Reduced perceived latency |
| HNSW index | Sub-linear query time at scale |

### 10.3 Features

- **Conversational memory** — include rolling chat history in the prompt for multi-turn Q&A.
- **Source attribution** — append `(Source: filename, page N)` to every retrieved passage in the answer.
- **Evaluation harness** — use [RAGAS](https://github.com/explodinggradients/ragas) to automatically score context relevance, answer faithfulness, and answer relevance.
- **Metadata filtering** — allow queries like "only search the Kubernetes PDF" by adding a pre-filter on the metadata before vector search.

---

## Project Structure

```
RAG/
├── src/
│   ├── app.py            # entry point
│   ├── data_loader.py    # multi-format document ingestion
│   ├── embeddings.py     # chunking + embedding pipeline
│   ├── vector_store.py   # FAISS build / persist / search
│   └── search.py         # end-to-end RAG query handler
├── data/
│   ├── pdf/              # source PDF documents
│   ├── text_files/       # source TXT documents
│   └── vector_store/     # legacy Chroma store (not used)
├── faiss_store/
│   ├── faiss.index       # serialised FAISS index
│   └── metadata.pkl      # chunk text metadata
├── .env                  # GROQ_API_KEY
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install langchain langchain-community langchain-groq \
            sentence-transformers faiss-cpu pypdf python-dotenv

# 2. Set your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Place documents in data/pdf/ or data/text_files/

# 4. Run
python src/app.py
```

---

*Documentation generated from implementation log and source code analysis.*
