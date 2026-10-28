"""Generate synthetic legal QA data and split it into tuning and test sets."""

# Create a synthetic question and reference answer pairs with an LLM using my corpus or chunks
# Validate those questions and answers pairs on groundedness, relevance and standalone
# Split 'em into tuning and held-out test set
# Tuning set to check for precision, context, faithfulness, relevance and correctness
# Test set for final testing

import json
import random
from tqdm import tqdm
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from rag.src.config import CORPUS_DIR
from rag.src.llm import llm
from ..prompts.eval_prompts import QA_generation_prompt
# from ..prompts.eval_prompts import QA_generation_prompt, QUESTION_VALIDATION_PROMPT, ANSWER_VALIDATION_PROMPT


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data"


def get_corpus_files(corpus_dir: str) -> list:
    """Get all corpus files in the corpus directory."""
    return list(Path(corpus_dir).rglob("*.jsonl"))


def read_corpus_file(file_path: str) -> list:
    """Read a corpus file and return its content in a list."""
    content = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content


def get_qa(context: str):
    pipe = QA_generation_prompt | llm | StrOutputParser()
    result = pipe.invoke({"context": context})
    return result


def sample_data(no_generations=10):
    # Gets corpus files
    corpus_files = get_corpus_files(corpus_dir=CORPUS_DIR)

    # Loads corpus records
    data = []
    for file in corpus_files:
        data.extend(read_corpus_file(file))
    
    # take 20 samples
    sample = random.sample(data, no_generations)
    outputs = []

    for sampled_context in tqdm(sample):
        try:
            qa_couple = get_qa(sampled_context["text"])
            
            question = qa_couple.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
            answer = qa_couple.split("Answer: ")[-1].strip()

            outputs.append({
                "source_doc": {
                    key: value
                    for key, value in sampled_context.items()
                    if key != "text"
                },
                "context": sampled_context["text"],
                "question": question,
                "answer": answer
            })

        except Exception as e:
            print(f"Error generating QA: {e}")

    return outputs


def save_samples(samples: list[dict]):
    """Save the first 10 samples for tuning and the next 10 for testing."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    tuning_samples = samples[:5]
    test_samples = samples[5:]

    datasets = [
        ("tuning.jsonl", tuning_samples),
        ("test.jsonl", test_samples),
    ]

    for filename, dataset in datasets:
        output_path = SAMPLES_DIR / filename

        with output_path.open("w", encoding="utf-8") as f:
            for sample in dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Saved {len(dataset)} samples to {output_path}.")


if __name__ == "__main__":
    samples = sample_data()
    save_samples(samples=samples)


# def validate_question(question: str, context: str):
#     """Evaluate a question's grounding, relevance, and standalone quality."""
#     pipe = QUESTION_VALIDATION_PROMPT | llm | StrOutputParser()
#     result = pipe.invoke({"question": question, "context": context})
#     return result


# def validate_answer(question: str, answer: str, context: str):
#     """Evaluate an answer's faithfulness, relevance, precision, and citations."""
#     pipe = ANSWER_VALIDATION_PROMPT | llm | StrOutputParser()
#     result = pipe.invoke({
#         "question": question,
#         "answer": answer,
#         "context": context,
#     })
#     return result


# def evaluate_sampled_data(samples: list[dict]):
#     """Return samples whose question and answer validations pass all thresholds."""
#     valid_samples = []

#     for sample in tqdm(samples):
#         question = sample["question"]
#         context = sample["context"]
#         answer = sample["answer"]

#         question_result = json.loads(
#             validate_question(question=question, context=context)
#             .strip()
#             .removeprefix("```json")
#             .removesuffix("```")
#             .strip()
#         )
#         answer_result = json.loads(
#             validate_answer(question=question, answer=answer, context=context)
#             .strip()
#             .removeprefix("```json")
#             .removesuffix("```")
#             .strip()
#         )

#         question_scores = (
#             "grounding_score",
#             "relevance_score",
#             "standalone_score",
#         )
#         question_passes = (
#             question_result.get("is_valid") is True
#             and all(question_result.get(key, 0) >= 4 for key in question_scores)
#         )
#         answer_scores = (
#             "faithfulness_score",
#             "relevance_score",
#             "legal_precision_score",
#             "citation_accuracy_score",
#         )
#         answer_passes = (
#             answer_result.get("is_valid") is True
#             and all(answer_result.get(key, 0) >= 4 for key in answer_scores)
#         )

#         if question_passes and answer_passes:
#             valid_samples.append({
#                 **sample,
#                 "question_validation": question_result,
#                 "answer_validation": answer_result,
#                 "is_valid": True,
#             })

#     return valid_samples
