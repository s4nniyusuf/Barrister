"""Compare a RAG answer with a reference answer from the tuning dataset."""

import json
from tqdm import tqdm

from .generate_qa_dataset import SAMPLES_DIR
from rag.src.pipeline import rag_pipeline


def main():
    tuning_path = SAMPLES_DIR / "tuning.jsonl"

    with tuning_path.open(encoding="utf-8") as file:
        tuning_samples = [json.loads(line) for line in file if line.strip()]
    
    for sample in tuning_samples:
        question = sample["question"]
        reference_answer = sample["answer"]
        rag_answer = rag_pipeline.invoke({"question": question, "history": []})

        print(f"Question:\n{question}\n")
        print(f"RAG answer:\n{rag_answer}\n")
        print(f"Reference answer:\n{reference_answer}")
        print(f"Reference Context: \n{sample["context"]}")
        print("\n")


if __name__ == "__main__":
    main()