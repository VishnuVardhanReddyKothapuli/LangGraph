# Folio

A small LangChain document chatbot: Qdrant hybrid search, Gemini/Groq fallback,
and a quiet beige-and-white interface. Start with
[`notebooks/01_understand_rag.ipynb`](notebooks/01_understand_rag.ipynb) to understand
each step before using the app.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this directory:

```powershell
uv sync --python 3.12 --extra dev
Copy-Item .env.example .env
# Edit .env: add GOOGLE_API_KEY, GROQ_API_KEY, or both.
uv run uvicorn folio.app:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. Upload `examples/team-handbook.md` and ask:
“What is the learning budget, and does a manager need to approve it?”

The first upload downloads FastEmbed models, so it takes longer and needs internet.
Later embeddings run locally. Qdrant persists under `data/qdrant`; no Docker is needed.
Keep one application worker with embedded Qdrant, and launch from the project root.
API keys are read by Python only. Set `PRIMARY_PROVIDER=groq` to reverse fallback order.
Restart after changing `.env`. Default models are configurable because availability changes.

Python 3.11–3.13 is supported. Without uv: create a Python 3.12 virtual environment,
activate it, and run `python -m pip install -e ".[dev]"`; use `python -m uvicorn`
and `python -m jupyterlab` in place of `uv run` commands below.

## Understand it in the notebook

```powershell
uv run python -m jupyterlab notebooks/01_understand_rag.ipynb
```

Choose the virtual environment's Python kernel and run the cells in order. The
notebook uses in-memory Qdrant, separate from the application library. It shows:

1. Loading text and metadata from documents.
2. Splitting into 350-token chunks with 60-token overlap.
3. Local dense embeddings and sparse BM25 embeddings.
4. Qdrant hybrid retrieval with reciprocal rank fusion and sparse IDF weights.
5. Packing evidence and recent conversation into the context budget.
6. Generating a cited answer, falling back to the other configured provider on failure.

The final generation cell skips cleanly when no key is configured. No outputs or
API credentials are saved in the committed notebook.

## Small, standard structure

```text
notebooks/01_understand_rag.ipynb   # explanation and runnable pipeline
src/folio/
  rag.py                          # loading, search, budget, models
  app.py                          # FastAPI endpoints
  static/                         # plain HTML, CSS, JavaScript
tests/test_rag.py                  # offline checks
examples/team-handbook.md          # a document to try
.env.example
pyproject.toml
uv.lock                           # reproducible dependency versions
```

## The 5,000-token window

The **application** budget is 5,000 reference tokens: at most 3,900 input tokens,
1,000 output tokens, and a 100-token formatting reserve. Input includes system
instructions, the question, source labels, evidence, and up to 700 tokens of recent
complete conversation turns. Questions above 1,000 reference tokens are rejected.
Older turns are discarded; retrieved excerpts are fitted to the remaining space.

Reference counts use `cl100k_base`. Gemini and Llama use different tokenizers, so
this is **not an exact provider-native 5,000-token cap**. Both providers separately
receive a 1,000-output-token limit. Gemini's thinking budget is disabled for the
default model so that it does not consume the answer allowance.

Hybrid search finds up to eight passages. It combines meaning-based dense search
with keyword-based BM25 search in Qdrant. Simple follow-ups add the previous user
question to the retrieval query; there is no extra LLM rewrite call. Source
drawers display the exact excerpts supplied to the model. Citations are generated
by the model, so inspect the excerpts when correctness matters.

## Storage and limits

- PDF with extractable text, UTF-8 TXT and Markdown; 10 MB/file, 200 PDF pages,
  and 1 million extracted characters. No OCR or DOCX support.
- English-oriented local embedding models. Changing embeddings requires a new
  collection and reindexing; do not point incompatible models at an existing collection.
- Re-uploading the same name and content is idempotent. Changed content is a new
  document version and does not delete old passages. Use distinct filenames for
  versions; set a fresh `QDRANT_COLLECTION` for a clean library.
- Documents persist across restarts; chat history and the uploaded-file list live
  only in the current browser tab. The sidebar's passage count includes the entire
  persisted library. “New conversation” resets messages, not documents.
- Only extracted chunks and metadata are stored in Qdrant, not original files.
  Relevant excerpts and recent conversation are sent to the chosen AI provider.
- This is a local, single-user app with clean project structure, validation,
  timeouts, bounded prompts, and safe text rendering. It has no authentication or
  tenant isolation. Before public/multi-user deployment, add authentication, isolate
  collections per user, enforce request/rate limits at a proxy, and use server Qdrant.
  Large PDF parsing should move to a resource-limited worker if accepting untrusted users.

For Qdrant Cloud or an existing server, set `QDRANT_URL` and `QDRANT_API_KEY` in
`.env`; omit `QDRANT_PATH`. Keep the single worker for this small app. No server
or cloud infrastructure is created automatically.

## Check it

```powershell
uv run python -m unittest discover -s tests -v
```

Checks cover prompt limits, ingestion validation, chunking, fallback in both
directions, provider failure, duplicate uploads, a real local Qdrant/API round
trip using deterministic test embeddings, and notebook validity. Live provider
success requires your credentials. The notebook exercises the real embedding models.

Implementation references: [Qdrant's LangChain hybrid integration](https://qdrant.tech/documentation/frameworks/langchain/),
[LangChain Gemini integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai),
and [Groq model catalog](https://console.groq.com/docs/models).
"# LangGraph" 
