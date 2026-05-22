# DEPI Graduation Project — Egyptian Law RAG Agent

A retrieval-augmented generation (RAG) agent that answers questions about **Egyptian law** using a legal corpus, multilingual embeddings, and an LLM routed through OpenRouter. The agent is implemented as a **LangGraph** pipeline with intent classification, retrieval, and Arabic answer generation.

---

## Features

- **Intent checker** — classifies questions as on-topic (Egyptian law) or off-topic before any retrieval
- **Vector store** — [Chroma](https://www.trychroma.com/) with cosine similarity, persisted under `./chroma_db`
- **Embeddings** — `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (Arabic + multilingual)
- **6-law corpus** — Constitution, Civil Code, Penal Code, Commercial Law, Criminal Procedures, Civil Procedures
- **Orchestration** — LangGraph pipeline with conditional routing
- **LLM** — Configured via LangChain's OpenRouter integration (model set in `rag_agent/utils/nodes.py`)
- **UI** — Streamlit chat interface with source document previews

---

## Graph Architecture

```
START
  └─► intent_check
        ├─► [on_topic]  → retrieve → generate → END
        └─► [off_topic] → reject              → END
```

---

## Repository Layout

| Path | Role |
|------|------|
| `app.py` | Streamlit web UI |
| `rag_agent/agent.py` | Builds and compiles the LangGraph |
| `rag_agent/utils/nodes.py` | Embeddings, Chroma retriever, prompts, LLM, all graph nodes |
| `rag_agent/utils/state.py` | `GraphState`: question, documents, answer, intent |
| `rag_agent/utils/ingest.py` | Loads all 6 PDFs, splits, embeds, writes `./chroma_db` |
| `documents/` | Source PDFs and `docs_references.txt` |
| `requirements.txt` | Python dependencies |
| `.env` | API keys (git-ignored) |

---

## Prerequisites

- Python **3.10+**
- An **[OpenRouter](https://openrouter.ai/)** API key

---

## Setup

1. **Clone** the repository and open a terminal in the `DEPI-Graduation-Project` folder.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key** — create a `.env` file:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```

---

## Ingesting Documents

Run once to build the vector store from all 6 legal PDFs:

```bash
python -m rag_agent.utils.ingest
```

This creates `./chroma_db`. Re-run only if you add or change source documents.

---

## Running the App

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**Terminal only:**
```bash
python -m rag_agent.agent
```

---

## Legal Corpus

| Law | File |
|-----|------|
| Egyptian Constitution 2019 | `دستور-جمهورية-مصر-العربية-2019.pdf` |
| Civil Code (Law 131/1948) | `القانون رقم 131 لسنة 1948 بإصدار القانون المدني.pdf` |
| Penal Code (Law 58/1937) | `قانون العقوبات رقم 58 لسنة 1937.pdf` |
| Commercial Law (Law 17/1999) | `قانون التجارة 17 لسنة 1999 وتعديلاته.pdf` |
| Criminal Procedures (Law 150/1950) | `قانون الإجراءات الجنائية رقم 150 لسنة 1950.pdf` |
| Civil Procedures (Law 13/1968) | `قانون رقم ۱۳ لسنة ۱۹٦۸ بإصدار قانون المرافعات المدنية والتجارية.pdf` |

---

## Disclaimer

This software is for **learning and research** purposes only. It does not constitute legal counsel. Always verify answers against official gazettes and qualified legal professionals.
