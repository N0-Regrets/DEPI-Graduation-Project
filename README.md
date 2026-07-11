# ⚖️ Arabic Legal RAG Agent

An Arabic-language **Retrieval-Augmented Generation (RAG)** assistant focused on **Egyptian law**. Users ask legal questions in Arabic (or English), and the system retrieves relevant passages from indexed legal documents, then generates grounded answers with source citations.

Built with **LangGraph**, **Chroma**, **HuggingFace embeddings**, and **Ollama**.

> **Disclaimer:** This tool provides general legal information for educational purposes only. It is **not** a substitute for professional legal advice.

---

## Supported Legal Documents

- Egyptian Constitution 
- Penal Code
- Commercial Law 
- Civil and Commercial Procedures

See `documents/docs_references.txt` for download links.

---



## Architecture

The agent is implemented as a **LangGraph state machine**:

```
User Question
     │
     ▼
┌─────────────┐
│ Intent Check│
└──────┬──────┘
       │
       ├── legal_question ──────► Retrieve (top-k=3) ──► Generate ──► Answer
       ├── greeting_or_meta ──────────────────────────► Generate ──► Answer
       ├── clarification_needed ──────────────────────► Clarify ──► Answer
       ├── out_of_scope ──────────────────────────────► Reject ──► Answer
       └── unsafe_or_disallowed ──────────────────────► Reject ──► Answer
```


| Component     | Technology                  |
| ------------- | --------------------------- |
| Embeddings    | `BAAI/bge-m3` (HuggingFace) |
| Vector store  | Chroma (`./chroma_db`)      |
| LLM           | Ollama (`gemma3:4b`)        |
| Orchestration | LangGraph                   |
| UI            | Streamlit (AI-generated)    |


The Streamlit chat interface in `app.py` was created with AI assistance. The RAG pipeline, LangGraph workflow, and evaluation logic were implemented separately.

---



## Prerequisites

- **Python 3.13**
- **[Ollama](https://ollama.com/)** installed and running locally
- The `gemma3:4b` model pulled in Ollama:

```bash
ollama pull gemma3:4b
```

---



## Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd "DEPI-Graduation-Project"
```

1. **Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

1. **Build the vector store** (first-time setup only)

```bash
python -m rag_agent.utils.ingest
```

This loads the PDFs, splits them into chunks (1024 tokens, 128 overlap), embeds them, and persists the index to `./chroma_db`.

---



## Usage

Start the Streamlit app:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). Type a legal question in Arabic or click one of the suggested sample questions.

---



## Evaluation

The project includes a DeepEval-based evaluation script that measures **faithfulness** and **contextual relevancy** against a set of 20 Arabic legal questions.

```bash
python -m rag_agent.tests.test_agent
```

---



## Future Work
- **Conversation memory**: retain prior turns in the LangGraph state to support natural follow-up questions.
- **Hybrid retrieval**: combine lexical (keyword/BM25) and semantic (embedding) search to improve recall on legal terms and article numbers
- **Better edge-case handling**: improve routing and responses for ambiguous questions, out-of-scope requests, unsafe prompts, and cases where retrieval returns insufficient context
- **Expanded evaluation set**: add more questions across all covered laws to better measure retrieval and answer quality

---



## Project Structure

```
.
├── app.py                          # Streamlit chat UI (AI-generated)
├── requirements.txt
├── documents/                      # Legal PDF source files
│   └── docs_references.txt         # Download links for legal texts
├── chroma_db/                      # Vector store (generated, gitignored)
└── rag_agent/
    ├── agent.py                    # LangGraph workflow definition
    ├── utils/
    │   ├── ingest.py               # PDF loading and vector store creation
    │   ├── nodes.py                # Graph nodes (intent, retrieve, generate)
    │   └── state.py                # Shared graph state schema
    └── tests/
        ├── test_agent.py           # DeepEval evaluation script
        └── results.json            # Cached Q&A results for evaluation
```

---



## Configuration


| Setting               | Location                    | Default        |
| --------------------- | --------------------------- | -------------- |
| LLM model             | `rag_agent/utils/nodes.py`  | `gemma3:4b`    |
| Embedding model       | `rag_agent/utils/nodes.py`  | `BAAI/bge-m3`  |
| Retrieval count (`k`) | `rag_agent/utils/nodes.py`  | `3`            |
| Chunk size / overlap  | `rag_agent/utils/ingest.py` | `1024` / `128` |
| Vector store path     | `rag_agent/utils/`          | `./chroma_db`  |


To re-index after adding or updating PDFs, delete `chroma_db/` and re-run the ingest script.