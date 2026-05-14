# GraphRAG Inference Benchmark

> **Proving Graph Beats Tokens** — A side-by-side comparison of LLM-Only, Basic RAG, and GraphRAG pipelines for the [GraphRAG Inference Hackathon by TigerGraph](https://unstop.com).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![TigerGraph](https://img.shields.io/badge/Graph-TigerGraph-orange?logo=data:image/png;base64,)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-4285F4?logo=google)

---

## 🎯 What This Project Does

This project benchmarks **three inference pipelines** on the same dataset and questions, proving that **GraphRAG delivers fewer tokens, faster responses, and maintained accuracy** compared to traditional RAG:

| Pipeline | How It Works | Token Usage |
|----------|-------------|-------------|
| **Pipeline 1: LLM-Only** | Raw question → Gemini → Answer | ❌ Highest (no context control) |
| **Pipeline 2: Basic RAG** | Question → ChromaDB vector search → chunk context → Gemini → Answer | ⚠️ High (big context dumps) |
| **Pipeline 3: GraphRAG** | Question → TigerGraph multi-hop reasoning → focused context → Gemini → Answer | ✅ Lowest (lean, relevant context) |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Your Dataset (≥2M tokens)                   │
│                  Wikipedia AI/ML Articles                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  User Query │
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          ▼                ▼                ▼
   ┌──────────┐    ┌──────────────┐   ┌──────────────┐
   │ LLM-Only │    │  Basic RAG   │   │   GraphRAG   │
   │ (Gemini) │    │(ChromaDB+LLM)│   │(TigerGraph   │
   │ No ctx   │    │ Vector search│   │ +LLM)        │
   └────┬─────┘    └──────┬───────┘   └──────┬───────┘
        │                 │                   │
        └────────┬────────┴───────┬───────────┘
                 ▼                ▼
        ┌────────────────────────────────┐
        │    Comparison Dashboard        │
        │  Tokens · Cost · Latency       │
        └────────────────┬───────────────┘
                         ▼
        ┌────────────────────────────────┐
        │  LLM-as-a-Judge + BERTScore   │
        │       Accuracy Evaluation      │
        └────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/graphrag-inference-benchmark.git
cd graphrag-inference-benchmark

# Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
# Edit .env with your API keys:
# - GEMINI_API_KEY (from https://aistudio.google.com/app/apikey)
# - TG_HOST, TG_USERNAME, TG_PASSWORD (from https://tgcloud.io)
```

### 3. Collect Dataset

```bash
python dataset/collector.py
# Downloads 300+ Wikipedia AI/ML articles (≥2M tokens)
```

### 4. Index for Basic RAG

```python
from pipelines.basic_rag import BasicRAGPipeline
p2 = BasicRAGPipeline()
p2.initialize()
p2.index_documents()
```

### 5. Setup GraphRAG (TigerGraph)

Follow the [TigerGraph GraphRAG setup guide](https://github.com/tigergraph/graphrag):

```bash
# Deploy GraphRAG via Docker
export LLM_API_KEY=your_gemini_key
curl -k https://raw.githubusercontent.com/tigergraph/graphrag/refs/heads/main/docs/tutorials/setup_graphrag.sh | bash
```

Then ingest your dataset:
```python
from pipelines.graphrag_pipeline import GraphRAGPipeline
p3 = GraphRAGPipeline()
p3.initialize()
p3.setup_graph()
p3.ingest_documents()
```

### 6. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### 7. Run Full Benchmark

```bash
python evaluation/benchmark.py
# Or with options:
python evaluation/benchmark.py --skip-graphrag  # Test without TigerGraph
python evaluation/benchmark.py --questions 3     # Quick test with 3 questions
```

## 📊 Dashboard Features

- **Live Query**: Enter one question → all 3 pipelines run → side-by-side results
- **Full Benchmark**: Automated evaluation with LLM-as-a-Judge + BERTScore
- **Past Results**: Browse and compare historical benchmark runs
- **Interactive Charts**: Plotly visualizations for tokens, latency, accuracy

## 📏 Evaluation Methods

### LLM-as-a-Judge
Uses Gemini to grade each answer **PASS/FAIL** based on factual accuracy, completeness, and relevance.

**Bonus threshold**: ≥ 90% pass rate

### BERTScore
Computes semantic similarity using DeBERTa model.

**Bonus thresholds**: F1 rescaled ≥ 0.55 | F1 raw ≥ 0.88

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Google Gemini 2.5 Flash | Free tier, fast, accurate |
| Vector DB | ChromaDB | Local, zero setup |
| Graph DB | TigerGraph Savanna | $60 free credits, managed |
| GraphRAG | TigerGraph GraphRAG repo | Pre-built knowledge graph + hybrid retrieval |
| Dashboard | Streamlit + Plotly | Python-native, interactive |
| Dataset | Wikipedia API | Public domain, entity-rich |
| Embeddings | Gemini text-embedding-004 | High quality, free tier |
| Evaluation | BERTScore + LLM Judge | HuggingFace, industry standard |

## 📁 Project Structure

```
graphinference/
├── config.py                  # Configuration & benchmark questions
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── dataset/
│   ├── collector.py           # Wikipedia data collector
│   └── data/                  # Downloaded articles
├── pipelines/
│   ├── base.py                # Base pipeline + metrics
│   ├── llm_only.py            # Pipeline 1: Direct LLM
│   ├── basic_rag.py           # Pipeline 2: ChromaDB + LLM
│   └── graphrag_pipeline.py   # Pipeline 3: TigerGraph + LLM
├── evaluation/
│   ├── llm_judge.py           # LLM-as-a-Judge (PASS/FAIL)
│   ├── bertscore_eval.py      # BERTScore semantic similarity
│   └── benchmark.py           # Full benchmark orchestrator
├── dashboard/
│   ├── app.py                 # Streamlit dashboard
│   ├── styles.py              # Premium dark theme CSS
│   └── components.py          # Reusable UI components
└── results/                   # Benchmark results (JSON)
```

## 🏆 Judging Criteria

| Criteria | Weight | Our Approach |
|----------|--------|-------------|
| Token Reduction | 30% | GraphRAG's focused context vs RAG's chunk dumps |
| Answer Accuracy | 30% | LLM-as-a-Judge + BERTScore dual evaluation |
| Performance | 20% | Latency benchmarks, efficient retrieval |
| Engineering & Storytelling | 20% | Clean architecture, interactive dashboard, this README |

## 📝 License

This project is built on top of the [TigerGraph GraphRAG repo](https://github.com/tigergraph/graphrag) (Apache-2.0).

---

*Built for the GraphRAG Inference Hackathon by TigerGraph* 🐯
*#GraphRAGInferenceHackathon*
