"""
Benchmark Runner
=================
Orchestrates running all 3 pipelines on the same questions,
then evaluates with LLM-as-a-Judge and BERTScore.
Saves comprehensive results for the dashboard.
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.llm_only import LLMOnlyPipeline
from pipelines.basic_rag import BasicRAGPipeline
from pipelines.graphrag_pipeline import GraphRAGPipeline
from evaluation.llm_judge import evaluate_batch as judge_batch
from evaluation.bertscore_eval import evaluate_pipeline_bertscore


def run_full_benchmark(questions: list[dict] = None, skip_graphrag: bool = False) -> dict:
    """
    Run the complete benchmark across all 3 pipelines.
    
    Args:
        questions: list of {"question": ..., "expected_answer": ...} dicts
        skip_graphrag: if True, skip Pipeline 3 (for testing without TigerGraph)
    
    Returns:
        Complete benchmark results dict
    """
    if questions is None:
        questions = config.BENCHMARK_QUESTIONS
    
    expected_answers = [q["expected_answer"] for q in questions]
    timestamp = datetime.now().isoformat()
    
    print("=" * 70)
    print("GRAPHRAG INFERENCE BENCHMARK")
    print(f"   Questions: {len(questions)}")
    print(f"   Timestamp: {timestamp}")
    print("=" * 70)
    
    results = {}
    
    # ===== Pipeline 1: LLM-Only =====
    print("\nPipeline 1: LLM-Only")
    print("-" * 40)
    p1 = LLMOnlyPipeline()
    p1_results = p1.run_benchmark(questions)
    results["llm_only"] = {
        "pipeline": "LLM-Only",
        "results": [r.to_dict() for r in p1_results],
        "avg_tokens": sum(r.total_tokens for r in p1_results) / len(p1_results),
        "avg_latency": sum(r.latency_seconds for r in p1_results) / len(p1_results),
        "avg_cost": sum(r.cost_usd for r in p1_results) / len(p1_results),
        "total_tokens": sum(r.total_tokens for r in p1_results),
    }
    
    # Small delay between pipelines to respect rate limits
    time.sleep(3)
    
    # ===== Pipeline 2: Basic RAG =====
    print("\nPipeline 2: Basic RAG")
    print("-" * 40)
    p2 = BasicRAGPipeline()
    p2_results = p2.run_benchmark(questions)
    results["basic_rag"] = {
        "pipeline": "Basic RAG",
        "results": [r.to_dict() for r in p2_results],
        "avg_tokens": sum(r.total_tokens for r in p2_results) / len(p2_results),
        "avg_latency": sum(r.latency_seconds for r in p2_results) / len(p2_results),
        "avg_cost": sum(r.cost_usd for r in p2_results) / len(p2_results),
        "total_tokens": sum(r.total_tokens for r in p2_results),
    }
    
    # Small delay between pipelines to respect rate limits
    time.sleep(3)
    
    # ===== Pipeline 3: GraphRAG =====
    if not skip_graphrag:
        print("\nPipeline 3: GraphRAG")
        print("-" * 40)
        p3 = GraphRAGPipeline()
        p3_results = p3.run_benchmark(questions)
        results["graphrag"] = {
            "pipeline": "GraphRAG",
            "results": [r.to_dict() for r in p3_results],
            "avg_tokens": sum(r.total_tokens for r in p3_results) / len(p3_results),
            "avg_latency": sum(r.latency_seconds for r in p3_results) / len(p3_results),
            "avg_cost": sum(r.cost_usd for r in p3_results) / len(p3_results),
            "total_tokens": sum(r.total_tokens for r in p3_results),
        }
    
    # Small delay before evaluation to respect rate limits
    time.sleep(3)
    
    # ===== Accuracy Evaluation =====
    print("\nRunning Accuracy Evaluation...")
    print("-" * 40)
    
    # LLM-as-a-Judge
    print("\nLLM-as-a-Judge Evaluation")
    results["judge"] = {}
    
    print("  Pipeline 1: LLM-Only")
    results["judge"]["llm_only"] = judge_batch(p1_results, expected_answers)
    
    time.sleep(3)
    
    print("  Pipeline 2: Basic RAG")
    results["judge"]["basic_rag"] = judge_batch(p2_results, expected_answers)
    
    if not skip_graphrag:
        time.sleep(3)
        print("  Pipeline 3: GraphRAG")
        results["judge"]["graphrag"] = judge_batch(p3_results, expected_answers)
    
    # BERTScore
    print("\nBERTScore Evaluation")
    results["bertscore"] = {}
    
    print("  Pipeline 1: LLM-Only")
    results["bertscore"]["llm_only"] = evaluate_pipeline_bertscore(p1_results, expected_answers)
    
    print("  Pipeline 2: Basic RAG")
    results["bertscore"]["basic_rag"] = evaluate_pipeline_bertscore(p2_results, expected_answers)
    
    if not skip_graphrag:
        print("  Pipeline 3: GraphRAG")
        results["bertscore"]["graphrag"] = evaluate_pipeline_bertscore(p3_results, expected_answers)
    
    # ===== Token Reduction Calculation =====
    print("\nToken Reduction Analysis")
    print("-" * 40)
    
    rag_tokens = results["basic_rag"]["avg_tokens"]
    llm_tokens = results["llm_only"]["avg_tokens"]
    
    results["comparison"] = {
        "llm_only_avg_tokens": llm_tokens,
        "basic_rag_avg_tokens": rag_tokens,
    }
    
    if not skip_graphrag:
        graphrag_tokens = results["graphrag"]["avg_tokens"]
        token_reduction_vs_rag = ((rag_tokens - graphrag_tokens) / rag_tokens * 100) if rag_tokens > 0 else 0
        token_reduction_vs_llm = ((llm_tokens - graphrag_tokens) / llm_tokens * 100) if llm_tokens > 0 else 0
        
        results["comparison"]["graphrag_avg_tokens"] = graphrag_tokens
        results["comparison"]["token_reduction_vs_rag_pct"] = round(token_reduction_vs_rag, 2)
        results["comparison"]["token_reduction_vs_llm_pct"] = round(token_reduction_vs_llm, 2)
        
        print(f"  GraphRAG vs Basic RAG: {token_reduction_vs_rag:.1f}% token reduction")
        print(f"  GraphRAG vs LLM-Only:  {token_reduction_vs_llm:.1f}% token reduction")
    
    # ===== Add metadata =====
    results["metadata"] = {
        "timestamp": timestamp,
        "num_questions": len(questions),
        "llm_model": config.LLM_MODEL,
        "rag_chunk_size": config.CHUNK_SIZE,
        "rag_top_k": config.RAG_TOP_K,
        "graphrag_method": config.GRAPHRAG_METHOD,
        "graphrag_top_k": config.GRAPHRAG_TOP_K,
        "graphrag_num_hops": config.GRAPHRAG_NUM_HOPS,
        "skip_graphrag": skip_graphrag,
    }
    
    # ===== Save Results =====
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(config.RESULTS_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to {filepath}")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    pipelines_data = [
        ("LLM-Only", results["llm_only"], results["judge"]["llm_only"], results["bertscore"]["llm_only"]),
        ("Basic RAG", results["basic_rag"], results["judge"]["basic_rag"], results["bertscore"]["basic_rag"]),
    ]
    if not skip_graphrag:
        pipelines_data.append(
            ("GraphRAG", results["graphrag"], results["judge"]["graphrag"], results["bertscore"]["graphrag"])
        )
    
    print(f"\n{'Pipeline':<15} {'Avg Tokens':<12} {'Avg Latency':<13} {'Pass Rate':<11} {'BERTScore F1':<14}")
    print("-" * 65)
    for name, pdata, jdata, bdata in pipelines_data:
        print(f"{name:<15} {pdata['avg_tokens']:<12.0f} {pdata['avg_latency']:<13.3f} {jdata['pass_rate']:<11.1%} {bdata['avg_f1_rescaled']:<14.4f}")
    
    print("=" * 70)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run GraphRAG Benchmark")
    parser.add_argument("--skip-graphrag", action="store_true", help="Skip GraphRAG pipeline (for testing)")
    parser.add_argument("--questions", type=int, default=None, help="Number of questions to run (default: all)")
    args = parser.parse_args()
    
    questions = config.BENCHMARK_QUESTIONS
    if args.questions:
        questions = questions[:args.questions]
    
    run_full_benchmark(questions=questions, skip_graphrag=args.skip_graphrag)
