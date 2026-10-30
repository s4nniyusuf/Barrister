"""Prompt templates for generating and evaluating synthetic legal QA data."""

from langchain_core.prompts import PromptTemplate


LLM_JUDGE_PROMPT = PromptTemplate.from_template("""
You are an expert AI system evaluator. Evaluate a Retrieval-Augmented Generation (RAG) output against the user question, retrieved context, and ground-truth reference answer.

Inputs:
- Question: {question}
- Retrieved Context: {retrieved_context}
- RAG Answer: {rag_answer}
- Reference Answer (Ground Truth): {reference_answer}

Score each metric from 0 to 5. Use fractional scores when helpful. Penalize unsupported or extrapolated claims in faithfulness.
1. faithfulness: proportion of claims in the RAG answer directly supported by retrieved context.
2. precision: whether retrieved passages are focused and relevant, without excessive noise.
3. recall: how much information needed to match the reference answer appears in retrieved context.
4. answer_relevance: whether the answer directly and concisely addresses the question.
5. answer_correctness: semantic and factual accuracy compared with the reference answer.

Analyze each metric carefully and give a brief rationale. Return only a valid JSON object matching this schema, with numeric scores from 0 to 5 and string rationales:
{{
  "faithfulness": {{"score": 0.0, "rationale": "..."}},
  "precision": {{"score": 0.0, "rationale": "..."}},
  "recall": {{"score": 0.0, "rationale": "..."}},
  "answer_relevance": {{"score": 0.0, "rationale": "..."}},
  "answer_correctness": {{"score": 0.0, "rationale": "..."}}
}}
""")


QA_generation_prompt = PromptTemplate.from_template("""
Your task is to generate a factoid question and a reference answer from the legal source provided.

Question requirements:
- Generate a realistic, concise question that a user might ask a Nigerian law information assistant.
- The question must be answerable using a specific factual detail in the source.
- Do not mention the passage, context, or source in the question.

Answer requirements:
- Answer the question accurately using only the source.
- Keep the answer concise and directly address the question.
- Cite the relevant law and section, subsection, or paragraph using the source's available identifying information.
- Do not invent or guess legal citations. 

Now here is the context.

Context: {context}\n
Output:::
""")


# QUESTION_VALIDATION_PROMPT = PromptTemplate.from_template("""
# You are validating a synthetic question for a Nigerian legal RAG system.

# Evaluate the question against these criteria, scoring each from 1 to 5:

# 1. GROUNDING: How clearly and unambiguously can the question be answered using the context?
# 2. RELEVANCE: How useful is the question for evaluating a Nigerian legal information assistant?
# 3. STANDALONE: How well can the question be understood without access to the context or preceding conversation?

# Output your evaluation in valid JSON format:

# {{
#   "rationale": "Briefly explain each score.",
#   "grounding_score": 1,
#   "relevance_score": 1,
#   "standalone_score": 1,
#   "overall_score": 1,
#   "is_valid": true
# }}

# Use integer scores from 1 to 5. Set is_valid to true only if all three criteria score at least 4. The overall_score should reflect the question's overall quality.

# Context:
# {context}

# Question:
# {question}

# Evaluation:
# """)


# ANSWER_VALIDATION_PROMPT = PromptTemplate.from_template("""
# You are validating a synthetic reference answer for a Nigerian legal RAG system.

# Evaluate the answer against these criteria, scoring each from 1 to 5:

# 1. FAITHFULNESS: How fully is every substantive legal claim in the answer supported by the context?
# 2. RELEVANCE: How accurately and directly does the answer address the question?
# 3. LEGAL PRECISION: How well does the answer preserve the relevant conditions, exceptions, qualifications, and limitations in the context?
# 4. CITATION ACCURACY: How accurately does the answer cite the relevant law and provision, where identifiable from the context?

# Output your evaluation in valid JSON format:

# {{
#   "rationale": "Briefly explain each score.",
#   "faithfulness_score": 1,
#   "relevance_score": 1,
#   "legal_precision_score": 1,
#   "citation_accuracy_score": 1,
#   "overall_score": 1,
#   "is_valid": true
# }}

# Use integer scores from 1 to 5. Set is_valid to true only if all four criteria score at least 4. The overall_score should reflect the answer's overall quality. Do not assume legal facts or citations that are not supported by the context.

# Context:
# {context}

# Question:
# {question}

# Reference Answer:
# {answer}

# Evaluation:
# """)
