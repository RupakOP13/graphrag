"""
Base Pipeline — Common Metrics Tracking
=========================================
Abstract base class for all three pipelines.
Tracks tokens, latency, cost, and provides a standardized result format.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from abc import ABC, abstractmethod

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


@dataclass
class PipelineResult:
    """Standardized result from any pipeline."""
    pipeline_name: str
    question: str
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    context_text: str = ""          # The context that was sent to the LLM
    context_tokens: int = 0         # Tokens in the context portion
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


def calculate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD based on Groq pricing."""
    if config.USE_FREE_TIER:
        return 0.0
    
    input_cost = (prompt_tokens / 1_000_000) * config.LLM_INPUT_COST_PER_1M
    output_cost = (completion_tokens / 1_000_000) * config.LLM_OUTPUT_COST_PER_1M
    return round(input_cost + output_cost, 8)


class BasePipeline(ABC):
    """Abstract base class for all pipelines."""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    @abstractmethod
    def initialize(self):
        """Initialize the pipeline (load models, connect to DBs, etc.)."""
        pass
    
    @abstractmethod
    def query(self, question: str) -> PipelineResult:
        """Run a question through the pipeline and return metrics."""
        pass
    
    def ensure_initialized(self):
        """Ensure pipeline is initialized before querying."""
        if not self._initialized:
            self.initialize()
            self._initialized = True
    
    def run_benchmark(self, questions: list[dict]) -> list[PipelineResult]:
        """Run a benchmark set of questions."""
        self.ensure_initialized()
        results = []
        
        for i, q in enumerate(questions):
            print(f"  [{self.name}] Question {i+1}/{len(questions)}: {q['question'][:60]}...")
            result = self.query(q["question"])
            result.metadata["expected_answer"] = q.get("expected_answer", "")
            result.metadata["question_index"] = i
            results.append(result)
            
            # Small delay between questions to respect Groq rate limits
            if i < len(questions) - 1:
                time.sleep(1.5)
        
        return results
