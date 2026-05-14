"""
Setup Helper — Quick project setup and verification
=====================================================
Run this after cloning to verify everything is configured correctly.
"""

import os
import sys

# Import config early to apply Windows Unicode fix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def check_dependencies():
    """Check if all required packages are installed."""
    print("📦 Checking dependencies...")
    packages = {
        "groq": "groq",
        "chromadb": "chromadb",
        "streamlit": "streamlit",
        "plotly": "plotly",
        "wikipediaapi": "wikipedia-api",
        "bert_score": "bert-score",
        "dotenv": "python-dotenv",
        "pandas": "pandas",
        "tqdm": "tqdm",
    }
    
    missing = []
    for module, pip_name in packages.items():
        try:
            __import__(module)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} — NOT INSTALLED")
            missing.append(pip_name)
    
    # Optional: pyTigerGraph
    try:
        __import__("pyTigerGraph")
        print(f"  ✅ pyTigerGraph")
    except ImportError:
        print(f"  ⚠️  pyTigerGraph — Not installed (needed for Pipeline 3)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies installed!")
    
    return len(missing) == 0


def check_config():
    """Verify configuration."""
    print("\n⚙️  Checking configuration...")
    
    import config
    
    # Groq API Key
    if config.GROQ_API_KEY and config.GROQ_API_KEY != "your_groq_api_key_here":
        print(f"  ✅ Groq API Key configured")
    else:
        print(f"  ❌ Groq API Key NOT configured — edit .env file")
    
    # TigerGraph
    if config.TG_HOST != "http://localhost":
        print(f"  ✅ TigerGraph host: {config.TG_HOST}")
    else:
        print(f"  ⚠️  TigerGraph using localhost (configure .env for Savanna)")
    
    # Directories
    print(f"  ✅ Dataset dir: {config.DATASET_DIR}")
    print(f"  ✅ Results dir: {config.RESULTS_DIR}")
    print(f"  ✅ ChromaDB dir: {config.CHROMA_PERSIST_DIR}")
    
    # Dataset
    dataset_file = os.path.join(config.DATASET_DIR, "wikipedia_ai_dataset.json")
    if os.path.exists(dataset_file):
        size_mb = os.path.getsize(dataset_file) / (1024 * 1024)
        print(f"  ✅ Dataset found ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  Dataset not collected yet — run: python dataset/collector.py")
    
    # ChromaDB index
    if os.path.exists(config.CHROMA_PERSIST_DIR) and os.listdir(config.CHROMA_PERSIST_DIR):
        print(f"  ✅ ChromaDB index found")
    else:
        print(f"  ⚠️  ChromaDB not indexed — will auto-index when Basic RAG runs")


def check_project_structure():
    """Verify all project files exist."""
    print("\n📂 Checking project structure...")
    
    required_files = [
        "config.py",
        "requirements.txt",
        ".env.example",
        "README.md",
        "dataset/__init__.py",
        "dataset/collector.py",
        "pipelines/__init__.py",
        "pipelines/base.py",
        "pipelines/llm_only.py",
        "pipelines/basic_rag.py",
        "pipelines/graphrag_pipeline.py",
        "evaluation/__init__.py",
        "evaluation/llm_judge.py",
        "evaluation/bertscore_eval.py",
        "evaluation/benchmark.py",
        "dashboard/__init__.py",
        "dashboard/app.py",
        "dashboard/styles.py",
        "dashboard/components.py",
    ]
    
    root = os.path.dirname(os.path.abspath(__file__))
    all_ok = True
    
    for f in required_files:
        path = os.path.join(root, f)
        if os.path.exists(path):
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} — MISSING")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick smoke test of Pipeline 1."""
    print("\n🧪 Running quick test (Pipeline 1: LLM-Only)...")
    
    try:
        import config
        if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
            print("  ⚠️  Skipping — Groq API Key not configured")
            return
        
        from pipelines.llm_only import LLMOnlyPipeline
        p1 = LLMOnlyPipeline()
        result = p1.query("What is artificial intelligence?")
        
        print(f"  ✅ Got response!")
        print(f"     Tokens: {result.total_tokens}")
        print(f"     Latency: {result.latency_seconds}s")
        print(f"     Answer: {result.answer[:100]}...")
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 GraphRAG Inference Benchmark — Setup Check")
    print("=" * 60)
    
    deps_ok = check_dependencies()
    check_config()
    structure_ok = check_project_structure()
    
    if deps_ok:
        run_quick_test()
    
    print("\n" + "=" * 60)
    if deps_ok and structure_ok:
        print("✅ Project is ready!")
        print("\nNext steps:")
        print("  1. Edit .env with your API keys")
        print("  2. python dataset/collector.py     (collect dataset)")
        print("  3. streamlit run dashboard/app.py  (launch dashboard)")
    else:
        print("⚠️  Some issues found — see above for details")
        print("  Run: pip install -r requirements.txt")
    print("=" * 60)
