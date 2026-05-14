"""
BERTScore Evaluation
=====================
Measures semantic similarity between predicted and expected answers.
Uses the bert_score library with DeBERTa model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_bertscore(predictions: list[str], references: list[str], model_type: str = None) -> dict:
    """
    Compute BERTScore between predicted and reference answers.
    
    Args:
        predictions: list of predicted answer strings
        references: list of reference/expected answer strings
        model_type: BERTScore model (default: config.BERTSCORE_MODEL)
    
    Returns:
        dict with precision, recall, f1 scores (per-sample and averaged)
    """
    from bert_score import score as bert_score
    
    if model_type is None:
        model_type = config.BERTSCORE_MODEL
    
    print(f"   Computing BERTScore with {model_type}...")
    
    # Try primary model, then fall back to lighter models if it fails
    # Note: roberta-large is the canonical bert_score model — deberta-xlarge-mnli is NLI-tuned
    models_to_try = [model_type, "distilbert-base-uncased"]
    if model_type != "roberta-large":
        models_to_try = [model_type, "roberta-large", "distilbert-base-uncased"]
    last_error = None
    
    precision_scores = recall_scores = f1_scores = f1_raw_list = None
    avg_precision = avg_recall = avg_f1 = avg_f1_raw = 0.0
    
    for attempt_model in models_to_try:
        try:
            print(f"   Trying BERTScore with {attempt_model}...")
            P, R, F1 = bert_score(
                predictions,
                references,
                model_type=attempt_model,
                lang="en",
                verbose=False,
                rescale_with_baseline=True
            )
            
            precision_scores = P.tolist()
            recall_scores = R.tolist()
            f1_scores = F1.tolist()
            
            avg_precision = sum(precision_scores) / len(precision_scores)
            avg_recall = sum(recall_scores) / len(recall_scores)
            avg_f1 = sum(f1_scores) / len(f1_scores)
            
            # Raw (non-rescaled) scores
            P_raw, R_raw, F1_raw = bert_score(
                predictions,
                references,
                model_type=attempt_model,
                lang="en",
                verbose=False,
                rescale_with_baseline=False
            )
            
            avg_f1_raw = sum(F1_raw.tolist()) / len(F1_raw.tolist())
            f1_raw_list = F1_raw.tolist()
            model_type = attempt_model  # Record which model actually worked
            print(f"   BERTScore computed successfully with {attempt_model}")
            break  # Success — stop trying
            
        except Exception as e:
            last_error = e
            print(f"   BERTScore with {attempt_model} failed: {type(e).__name__}: {e}")
            continue
    
    if precision_scores is None:
        # All models failed
        print(f"   All BERTScore models failed. Last error: {last_error}")
        print(f"   Run: venv\\Scripts\\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu")
        length = len(predictions)
        precision_scores = [0.0] * length
        recall_scores = [0.0] * length
        f1_scores = [0.0] * length
        f1_raw_list = [0.0] * length

    return {
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "f1_scores": f1_scores,
        "f1_raw_scores": f1_raw_list,
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_f1_rescaled": round(avg_f1, 4),
        "avg_f1_raw": round(avg_f1_raw, 4),
        "model": model_type,
        "meets_bonus_f1_rescaled": avg_f1 >= 0.55,   # Bonus: ≥0.55
        "meets_bonus_f1_raw": avg_f1_raw >= 0.88,      # Bonus: ≥0.88
    }


def evaluate_pipeline_bertscore(results: list, expected_answers: list) -> dict:
    """
    Evaluate a pipeline's answers using BERTScore.
    
    Args:
        results: list of PipelineResult objects
        expected_answers: list of expected answer strings
    
    Returns:
        dict with BERTScore metrics
    """
    predictions = [r.answer for r in results]
    
    scores = compute_bertscore(predictions, expected_answers)
    scores["pipeline"] = results[0].pipeline_name if results else "Unknown"
    scores["total_questions"] = len(results)
    
    # Per-question breakdown
    per_question = []
    for i, (result, expected) in enumerate(zip(results, expected_answers)):
        per_question.append({
            "question_index": i,
            "question": result.question[:100],
            "f1_rescaled": round(scores["f1_scores"][i], 4),
            "f1_raw": round(scores["f1_raw_scores"][i], 4),
            "precision": round(scores["precision_scores"][i], 4),
            "recall": round(scores["recall_scores"][i], 4),
        })
    
    scores["per_question"] = per_question
    return scores
