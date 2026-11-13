"""Offline regression tests: python -m unittest discover -s tests -v."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class HistoryRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.model_calls = []
        self.queries = []

        def model(prompt):
            self.model_calls.append(prompt)
            return AIMessage(content="What are the exceptions under Lagos Tenancy Law?")

        def retrieve(question):
            self.queries.append(question)
            return ["Relevant provision"]

        dependencies = {}
        for name, attrs in {
            "rag.src.llm": {"llm": RunnableLambda(model)},
            "rag.src.hybrid_retriever": {"retriever": RunnableLambda(retrieve)},
            "rag.src.utils": {
                "select_top_docs": lambda docs: docs[:10],
                "format_docs": lambda docs: "\n".join(docs),
            },
        }.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            dependencies[name] = module
        path = Path(__file__).resolve().parents[1] / "rag/src/pipeline.py"
        spec = importlib.util.spec_from_file_location("rag.src._history_test_pipeline", path)
        self.pipeline = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, dependencies):
            spec.loader.exec_module(self.pipeline)

    def test_first_question_skips_contextualization(self):
        question = "What does Lagos Tenancy Law cover?"
        self.assertEqual(self.pipeline.retrieval_question({"question": question}), question)
        self.assertEqual(self.model_calls, [])

    def test_follow_up_retrieves_standalone_question_but_answers_original(self):
        question = "What are the exceptions?"
        history = [{"role": "user", "content": "What does Lagos Tenancy Law cover?"}]
        self.pipeline.rag_pipeline.invoke({"question": question, "history": history})
        self.assertEqual(self.queries, ["What are the exceptions under Lagos Tenancy Law?"])
        rewrite_prompt = self.model_calls[0].to_string()
        answer_prompt = self.model_calls[1].to_string()
        self.assertIn(history[0]["content"], rewrite_prompt)
        self.assertIn("Question:\n" + question, answer_prompt)
        self.assertIn("Relevant provision", answer_prompt)

    def test_blank_rewrite_falls_back_to_original(self):
        with patch.object(self.pipeline, "contextualize_question", RunnableLambda(lambda _: "  ")):
            self.assertEqual(self.pipeline.retrieval_question({
                "question": "Exceptions?",
                "history": [{"role": "user", "content": "Lagos tenancy"}],
            }), "Exceptions?")

    def test_history_is_limited_to_recent_twelve_messages(self):
        history = [{"role": "user", "content": f"turn-{i}"} for i in range(20)]
        self.pipeline.retrieval_question({"question": "Exceptions?", "history": history})
        prompt = self.model_calls[0].to_string()
        self.assertNotIn("User: turn-7\n", prompt)
        self.assertIn("User: turn-8\n", prompt)
        self.assertIn("User: turn-19", prompt)


if __name__ == "__main__":
    unittest.main()
