"""Offline regression tests: python -m unittest discover -s tests -v."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


def load_module(name, relative_path, dependencies):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def stub(name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    return module


class EvaluationContextTests(unittest.TestCase):
    def setUp(self):
        self.queries = []
        self.generation_contexts = []

        def retrieve(question):
            self.queries.append(question)
            # A second retrieval deliberately returns a different context.
            return [f"provision from retrieval {len(self.queries)}"]

        def generate(prompt):
            self.generation_contexts.append(prompt.to_string())
            return AIMessage(content="Grounded answer")

        self.pipeline = load_module("rag.src._evaluation_test_pipeline", "rag/src/pipeline.py", {
            "rag.src.llm": stub("rag.src.llm", llm=RunnableLambda(generate)),
            "rag.src.hybrid_retriever": stub("rag.src.hybrid_retriever", retriever=RunnableLambda(retrieve)),
            "rag.src.utils": stub("rag.src.utils", select_top_docs=lambda docs: docs[:10],
                                  format_docs=lambda docs: "\n".join(docs)),
        })

    def test_returns_exact_generation_context_with_single_retrieval(self):
        response = self.pipeline.rag_pipeline_with_context.invoke({"question": "A question", "history": []})
        self.assertEqual(self.queries, ["A question"])
        self.assertEqual(response["answer"], "Grounded answer")
        self.assertEqual(response["context"], "provision from retrieval 1")
        self.assertIn(response["context"], self.generation_contexts[0])

    def test_application_pipeline_still_returns_string(self):
        self.assertEqual(self.pipeline.rag_pipeline.invoke({"question": "A question"}), "Grounded answer")
        self.assertEqual(len(self.queries), 1)

    def test_evaluator_judges_context_used_for_generation(self):
        judged = []

        def judge(values):
            judged.append(values)
            return '{"faithfulness": {"score": 5}}'

        with tempfile.TemporaryDirectory() as directory:
            samples_dir = Path(directory)
            sample = {"question": "A question", "answer": "Reference", "context": "Reference context"}
            (samples_dir / "tuning.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            evaluator = load_module("evaluation.src._test_evaluate", "evaluation/src/evaluate.py", {
                "evaluation.src.generate_qa_dataset": stub("evaluation.src.generate_qa_dataset", SAMPLES_DIR=samples_dir),
                "evaluation.src.llm_as_a_judge": stub("evaluation.src.llm_as_a_judge", llm_as_a_judge=RunnableLambda(judge)),
                "rag.src.pipeline": self.pipeline,
            })
            with redirect_stdout(io.StringIO()):
                evaluator.main()
            results = json.loads((samples_dir / "evaluation_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(self.queries), 1)
        self.assertEqual(judged[0]["retrieved_context"], "provision from retrieval 1")
        self.assertIn(judged[0]["retrieved_context"], self.generation_contexts[0])
        self.assertEqual(judged[0]["rag_answer"], "Grounded answer")
        self.assertEqual(results[-1]["average_scores"]["faithfulness"], 5)


if __name__ == "__main__":
    unittest.main()
