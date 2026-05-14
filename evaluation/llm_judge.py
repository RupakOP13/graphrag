"""
LLM-as-a-Judge Evaluation
===========================
Uses an LLM to grade answers PASS/FAIL.

Strategy:
  1. If HF_TOKEN is set → uses HuggingFace Inference API (InferenceClient)
  2. Fallback → uses Groq (Llama 3.1 8B Instant, a HuggingFace-hosted model)

Both options use open-source HuggingFace models for evaluation.
"""

import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _build_judge_prompt(question: str, predicted_answer: str, expected_answer: str) -> str:
    return f"""You are an expert evaluator. Judge whether a predicted answer is correct compared to the expected answer.

QUESTION: {question}

EXPECTED ANSWER: {expected_answer}

PREDICTED ANSWER: {predicted_answer}

Evaluate on these criteria:
1. Factual Accuracy: Are the facts correct?
2. Completeness: Does it cover the key points?
3. Relevance: Does it address the question?

Rules:
- PASS if the predicted answer is largely correct and covers most key points.
- FAIL if it has major factual errors or misses most key points.
- Minor omissions or different wording is OK for PASS.

Respond in EXACTLY this format:
VERDICT: [PASS or FAIL]
REASONING: [One paragraph explanation]
CONFIDENCE: [HIGH, MEDIUM, or LOW]"""


def _parse_verdict(text: str) -> dict:
    """Parse the judge's response into structured output."""
    verdict = "FAIL"
    if re.search(r"VERDICT:\s*PASS", text, re.IGNORECASE):
        verdict = "PASS"
    elif re.search(r"VERDICT:\s*FAIL", text, re.IGNORECASE):
        verdict = "FAIL"

    reasoning = text
    if "REASONING:" in text.upper():
        parts = re.split(r"REASONING:", text, flags=re.IGNORECASE)
        reasoning = parts[-1].split("CONFIDENCE:")[0].strip() if len(parts) > 1 else text

    confidence = "MEDIUM"
    if re.search(r"CONFIDENCE:\s*HIGH", text, re.IGNORECASE):
        confidence = "HIGH"
    elif re.search(r"CONFIDENCE:\s*LOW", text, re.IGNORECASE):
        confidence = "LOW"

    return {
        "verdict": verdict,
        "score": 1.0 if verdict == "PASS" else 0.0,
        "reasoning": reasoning[:500],
        "confidence": confidence,
        "raw_response": text[:800]
    }


def _judge_via_huggingface(prompt: str) -> str:
    """Call HuggingFace Inference API using the huggingface_hub InferenceClient."""
    hf_token = getattr(config, 'HF_TOKEN', '') or ''
    if not hf_token:
        return ""

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=hf_token)

        # Try text_generation with a free model
        response = client.text_generation(
            prompt=prompt,
            model="HuggingFaceH4/zephyr-7b-beta",
            max_new_tokens=300,
            temperature=0.01,
            do_sample=False,
        )
        if response:
            print(f"  [HF] Got response from HuggingFace Inference API")
            return response
    except Exception as e:
        print(f"  [HF] Inference failed: {type(e).__name__}: {str(e)[:100]}")

    return ""


def _judge_via_groq(prompt: str) -> str:
    """Use Groq with Llama 3.1 8B (an open-source HuggingFace model)."""
    try:
        from groq import Groq
        client = Groq(api_key=config.GROQ_API_KEY)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=config.LLM_FAST_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a strict but fair answer evaluator. Always respond in the exact format requested."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait = 10 * (attempt + 1)
                    try:
                        match = re.search(r"try again in (\d+\.?\d*)s", error_str, re.IGNORECASE)
                        if match:
                            wait = float(match.group(1)) + 1
                    except Exception:
                        pass
                    print(f"  [Groq] Rate limit, waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise
    except Exception as e:
        print(f"  [Groq] Judge failed: {e}")
    return ""


def judge_answer(question: str, predicted_answer: str, expected_answer: str) -> dict:
    """
    Use LLM as a judge to evaluate answer quality (PASS/FAIL).

    Strategy:
      1. Try HuggingFace Inference API (if HF_TOKEN is configured)
      2. Fall back to Groq (Llama 3.1 8B — an open-source HuggingFace model)

    Returns:
        dict with 'verdict', 'score', 'reasoning', 'judge_source'
    """
    prompt = _build_judge_prompt(question, predicted_answer, expected_answer)

    # Try HuggingFace first (if token is set)
    hf_text = _judge_via_huggingface(prompt)
    if hf_text and ("VERDICT" in hf_text.upper() or "PASS" in hf_text.upper() or "FAIL" in hf_text.upper()):
        result = _parse_verdict(hf_text)
        result["judge_source"] = "huggingface_inference_api"
        result["judge_model"] = "HuggingFaceH4/zephyr-7b-beta"
        return result

    # Use Groq (Llama 3.1 8B — an open-source HuggingFace model)
    groq_text = _judge_via_groq(prompt)
    if groq_text:
        result = _parse_verdict(groq_text)
        result["judge_source"] = "groq"
        result["judge_model"] = config.LLM_FAST_MODEL
        return result

    # Both failed
    return {
        "verdict": "ERROR",
        "score": 0.0,
        "reasoning": "Both HuggingFace and Groq judge failed.",
        "confidence": "NONE",
        "raw_response": "",
        "judge_source": "none",
        "judge_model": "none"
    }


def evaluate_batch(results: list, expected_answers: list) -> dict:
    """
    Evaluate a batch of pipeline results.

    Args:
        results: list of PipelineResult objects
        expected_answers: list of expected answer strings

    Returns:
        dict with overall metrics and per-question judgments
    """
    judgments = []
    pass_count = 0

    for i, (result, expected) in enumerate(zip(results, expected_answers)):
        print(f"   Judging Q{i+1}/{len(results)}: {result.question[:50]}...")
        judgment = judge_answer(result.question, result.answer, expected)
        judgment["question_index"] = i
        judgment["question"] = result.question
        judgment["pipeline"] = result.pipeline_name
        judgments.append(judgment)

        if judgment["verdict"] == "PASS":
            pass_count += 1

        # Rate limiting delay between questions
        if i < len(results) - 1:
            time.sleep(2.0)

    total = len(judgments)
    pass_rate = pass_count / total if total > 0 else 0.0

    return {
        "pipeline": results[0].pipeline_name if results else "Unknown",
        "total_questions": total,
        "pass_count": pass_count,
        "fail_count": total - pass_count,
        "pass_rate": round(pass_rate, 4),
        "meets_bonus_threshold": pass_rate >= 0.90,   # >=90% for bonus
        "judgments": judgments
    }
