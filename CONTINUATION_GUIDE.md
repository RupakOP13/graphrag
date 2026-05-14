# Continuation Guide for GraphRAG Hackathon

This file is for any AI assistant picking up this project. All core code has been written.

## 🏁 Current State
- **Workspace**: `c:\graphinference`
- **Environment**: Python 3.11+ venv created at `.\venv`. Core dependencies installed.
- **Pipelines**: All 3 pipelines (LLM, Basic RAG, GraphRAG) are implemented.
- **Evaluation**: BERTScore and LLM-as-a-Judge are implemented.
- **UI**: Streamlit dashboard with premium styling is ready.

## 🚀 How to Proceed (Phase 1: Verification & Data)

1. **Verify Environment**:
   Run `.\venv\Scripts\python.exe setup_check.py` to ensure all modules are found.

2. **Collect Dataset**:
   Run `.\venv\Scripts\python.exe dataset/collector.py`.
   *Goal: Download ~300 Wikipedia articles to hit the 2M token requirement.*

3. **Populate Vector DB (Basic RAG)**:
   In a Python script/REPL:
   ```python
   from pipelines.basic_rag import BasicRAGPipeline
   p2 = BasicRAGPipeline()
   p2.initialize()
   p2.index_documents() # Uses the JSON from the collector
   ```

4. **TigerGraph Integration**:
   The user needs to provide a running TigerGraph GraphRAG service. Once available:
   - Update `.env` with credentials.
   - Run the ingestion logic in `pipelines/graphrag_pipeline.py`.

5. **Run Benchmark**:
   Run `.\venv\Scripts\python.exe evaluation/benchmark.py --skip-graphrag` (to test without TG) or without the flag to test all three.

6. **Launch Dashboard**:
   Run `.\venv\Scripts\streamlit.exe run dashboard/app.py`.

## 📌 Critical Files
- `config.py`: Contains benchmark questions and API settings.
- `evaluation/benchmark.py`: The "brain" that runs the side-by-side comparison.
- `dashboard/app.py`: The visual output for judges.
