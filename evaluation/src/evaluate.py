"""Compare RAG answers with reference answers from the tuning dataset."""

import json
from tqdm import tqdm

from .generate_qa_dataset import SAMPLES_DIR
from .llm_as_a_judge import llm_as_a_judge
from rag.src.hybrid_retriever import retriever
from rag.src.pipeline import rag_pipeline
from rag.src.utils import format_docs, select_top_docs


def main():
    tuning_path = SAMPLES_DIR / "tuning.jsonl"
    output_path = SAMPLES_DIR / "evaluation_results.json"

    with tuning_path.open(encoding="utf-8") as file:
        tuning_samples = [
            json.loads(line) for line in file if line.strip()
        ]

    results = []

    for sample in tqdm(tuning_samples):
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

        # Save each sample with the generated answer, retrieved context, and judge result.
        results.append({
            **sample,
            "retrieved_context": retrieved_context,
            "rag_answer": rag_answer,
            "evaluation": evaluation,
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2, default=str)

        print(f"Question:\n{question}\n")
        print(f"RAG answer:\n{rag_answer}\n")
        print(f"Reference answer:\n{reference_answer}\n")
        print(f"Retrieved context:\n{retrieved_context}\n")
        print(f"Reference context:\n{sample['context']}\n")
        print(f"LLM judge evaluation:\n{evaluation}")
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
