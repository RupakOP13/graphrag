"""
Pipeline 1: LLM-Only
=====================
Sends the question directly to Groq (Llama 3.3 70B) with no retrieval context.
This is the worst-case baseline -- maximum token burn, relying solely on LLM memory.
"""

import time
from groq import Groq

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipelines.base import BasePipeline, PipelineResult, calculate_cost


def _call_groq_with_retry(client, model, messages, temperature=0.1, max_tokens=1024, max_retries=3):
    """Call Groq API with automatic retry on rate limit (429) errors."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                # Extract wait time from error message if available
                wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                try:
                    import re
                    match = re.search(r"try again in (\d+\.?\d*)s", error_str, re.IGNORECASE)
                    if match:
                        wait_time = float(match.group(1)) + 1
                except Exception:
                    pass
                print(f"  [Rate limit hit] Waiting {wait_time:.0f}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise  # Non-rate-limit error, re-raise immediately
    # Final attempt (no catch)
    return client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )


class LLMOnlyPipeline(BasePipeline):
    """Pipeline 1: Direct LLM query with no retrieval."""
    
    def __init__(self):
        super().__init__("LLM-Only")
        self.client = None
    
    def initialize(self):
        """Initialize Groq client."""
        self.client = Groq(api_key=config.GROQ_API_KEY)
        print(f"LLM-Only pipeline initialized (model: {config.LLM_MODEL})")
    
    def query(self, question: str) -> PipelineResult:
        """Send question directly to LLM with no context."""
        self.ensure_initialized()
        
        start_time = time.time()
        
        try:
            response = _call_groq_with_retry(
                self.client,
                config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a knowledgeable AI assistant. Answer questions accurately and comprehensively based on your training data."},
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            
            latency = time.time() - start_time
            answer = response.choices[0].message.content
            
            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else len(question) // 4
            completion_tokens = usage.completion_tokens if usage else len(answer) // 4
            total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens
            
            return PipelineResult(
                pipeline_name=self.name,
                question=question,
                answer=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=round(latency, 3),
                cost_usd=calculate_cost(prompt_tokens, completion_tokens),
                context_text="",  # No context in LLM-only
                context_tokens=0,
                metadata={"model": config.LLM_MODEL}
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return PipelineResult(
                pipeline_name=self.name,
                question=question,
                answer=f"Error: {str(e)}",
                latency_seconds=round(latency, 3),
                metadata={"error": str(e), "model": config.LLM_MODEL}
            )
