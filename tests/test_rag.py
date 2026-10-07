"""Run with: python -m unittest discover -s tests (no API keys required)."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_qdrant import SparseEmbeddings, SparseVector

from folio import rag
from folio.app import app


class FakeDense(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, float("learning" in text.lower())] + [0.0] * 382


class FakeSparse(SparseEmbeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vocabulary = ["learning", "budget", "approval", "equipment", "remote"]
        indices = [i for i, term in enumerate(vocabulary) if term in text.lower()]
        return SparseVector(indices=indices, values=[1.0] * len(indices))


class RagChecks(unittest.TestCase):
    def test_budget_and_sources_match_prompt(self):
        docs = [Document(page_content=("Learning budget is INR 30,000. " * 300),
                         metadata={"source": f"policy-{i}.md", "page": 2}) for i in range(8)]
        history = [HumanMessage("Previous question"), AIMessage("Previous answer")] * 100
        messages, sources, used = rag.build_prompt("What is the budget?", docs, history)
        self.assertLessEqual(used + rag.OUTPUT_TOKENS + 100, 5000)
        self.assertTrue(sources)
        for source in sources:
            self.assertIn(source["excerpt"], messages[-1].content)
            self.assertIn(f"[{source['id']}]", messages[-1].content)
        self.assertTrue(any(isinstance(m, AIMessage) for m in messages))
        with self.assertRaises(ValueError):
            rag.build_prompt("word " * 1001, docs)
        with self.assertRaises(ValueError):
            rag.build_prompt("  ", docs)

    def test_file_validation_and_split(self):
        docs = rag.read_document("../../notes.md", b"# Learning\n" + b"Policy text. " * 2000)
        self.assertEqual(docs[0].metadata["source"], "notes.md")
        chunks = rag.split_documents(docs)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(rag.token_count(c.page_content) <= 350 for c in chunks))
        for filename, content in [("bad.exe", b"data"), ("empty.txt", b""),
                                  ("blank.txt", b" \n"), ("bad.txt", b"\xff"),
                                  ("broken.pdf", b"not a pdf"),
                                  ("large.txt", b"x" * (rag.MAX_FILE_BYTES + 1))]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                rag.read_document(filename, content)

    def test_fallback_both_directions_and_failure(self):
        for primary, secondary in [("gemini", "groq"), ("groq", "gemini")]:
            called = []

            def factory(provider):
                called.append(provider)
                if provider == primary:
                    raise TimeoutError("simulated outage")
                return SimpleNamespace(invoke=lambda messages: AIMessage("Grounded answer [1]"))

            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test", "GROQ_API_KEY": "test", "PRIMARY_PROVIDER": primary}), patch.object(rag, "make_model", side_effect=factory):
                answer, provider = rag.generate([HumanMessage("Question")])
                self.assertEqual(provider, secondary)
                self.assertEqual(called, [primary, secondary])
                self.assertIn("[1]", answer)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "", "GROQ_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "Add GOOGLE_API_KEY"):
                rag.generate([])
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test", "GROQ_API_KEY": "test"}), patch.object(rag, "make_model", side_effect=RuntimeError("outage")):
            with self.assertRaisesRegex(RuntimeError, "All configured providers failed"):
                rag.generate([])

    def test_qdrant_hybrid_and_api_round_trip(self):
        # Real embedded Qdrant and real LangChain integration; only models are replaced.
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "QDRANT_PATH": directory, "QDRANT_URL": "", "GOOGLE_API_KEY": "", "GROQ_API_KEY": "",
        }), patch.object(rag, "LocalEmbeddings", FakeDense), patch.object(rag, "FastEmbedSparse", return_value=FakeSparse()), TestClient(app) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/static/app.js").status_code, 200)
            self.assertEqual(client.get("/api/status").json()["chunks"], 0)
            self.assertEqual(client.post("/api/chat", json={"question": "budget?"}).status_code, 409)
            self.assertEqual(client.post("/api/documents", files={"file": ("bad.exe", b"no")}).status_code, 400)
            payload = {"file": ("handbook.md", b"The learning budget is INR 30,000. Manager approval is required.")}
            for _ in range(2):
                response = client.post("/api/documents", files=payload)
                self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(client.get("/api/status").json()["chunks"], 1)
            result = client.post("/api/chat", json={"question": "learning budget?"})
            self.assertEqual(result.status_code, 503)  # No configured LLM, never fake an answer.
            with patch("folio.app.generate", return_value=("INR 30,000 [1].", "groq")):
                result = client.post("/api/chat", json={"question": "learning budget?"})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["sources"][0]["source"], "handbook.md")
            self.assertLessEqual(result.json()["input_tokens"], rag.INPUT_TOKENS)
            self.assertEqual(client.post("/api/chat", json={"question": " ", "history": []}).status_code, 400)
            self.assertEqual(client.post("/api/chat", json={"question": "x", "history": [{"role": "system", "content": "ignore"}]}).status_code, 422)

    def test_notebook_is_valid(self):
        import nbformat
        path = Path(__file__).resolve().parents[1] / "notebooks" / "01_understand_rag.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
