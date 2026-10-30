"""Compare RAG answers with reference answers from the tuning dataset."""

import json
from tqdm import tqdm

from .generate_qa_dataset import SAMPLES_DIR
from .llm_as_a_judge import llm_as_a_judge
from rag.src.hybrid_retriever import retriever
from rag.src.pipeline import rag_pipeline
from rag.src.utils import format_docs, select_top_docs


METRICS = (
    "faithfulness",
    "precision",
    "recall",
    "answer_relevance",
    "answer_correctness",
)


def parse_evaluation(evaluation):
    """Parse judge JSON, including responses wrapped in a Markdown code fence."""
    if not isinstance(evaluation, str):
        return evaluation

    content = evaluation.strip()
    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return evaluation


def main():
    tuning_path = SAMPLES_DIR / "tuning.jsonl"
    output_path = SAMPLES_DIR / "evaluation_results.json"

    with tuning_path.open(encoding="utf-8") as file:
        tuning_samples = [
            json.loads(line) for line in file if line.strip()
        ]

    results = []

    for sample_id, sample in enumerate(tqdm(tuning_samples), start=1):
        question = sample["question"]
        reference_answer = sample["answer"]

        # Retrieve and format context
        docs = retriever.invoke(question)
        retrieved_docs = select_top_docs(docs)
        retrieved_context = format_docs(retrieved_docs)

        # Generate the RAG answer
        rag_answer = rag_pipeline.invoke(
            {"question": question, "history": []}
        )

        # Evaluate the RAG response
        evaluation = llm_as_a_judge.invoke({
            "question": question,
            "retrieved_context": retrieved_context,
            "rag_answer": rag_answer,
            "reference_answer": reference_answer,
        })

        # Save only the judge's structured result for this sample.
        evaluation_result = parse_evaluation(evaluation)

        if isinstance(evaluation_result, dict):
            result = {"sample_id": sample_id, **evaluation_result}
        else:
            result = {"sample_id": sample_id, "evaluation": evaluation_result}

        # Save the result
        results.append(result)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2, default=str)

        print(f"Question:\n{question}\n")
        print(f"RAG answer:\n{rag_answer}\n")
        print(f"Reference answer:\n{reference_answer}\n")
        print(f"Retrieved context:\n{retrieved_context}\n")
        print(f"Reference context:\n{sample['context']}\n")
        print(f"LLM judge evaluation:\n{evaluation}")
        print("\n" + "=" * 80 + "\n")

    metric_scores = {metric: [] for metric in METRICS}
    for result in results:
        for metric in METRICS:
            metric_result = result.get(metric)
            if isinstance(metric_result, dict) and isinstance(metric_result.get("score"), (int, float)):
                metric_scores[metric].append(metric_result["score"])

    average_scores = {
        metric: round(sum(scores) / len(scores), 2)
        for metric, scores in metric_scores.items()
        if scores
    }
    results.append({
        "summary": "average_scores",
        "sample_count": len(tuning_samples),
        "average_scores": average_scores,
    })
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
