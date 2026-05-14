"""
GraphRAG Inference Benchmark Dashboard
========================================
Interactive Streamlit dashboard — one query in, three pipelines run,
side-by-side responses + metrics out. The heart of the hackathon project.

Run: streamlit run dashboard/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dashboard.styles import CUSTOM_CSS
from dashboard.components import (
    render_header, render_metric_card, render_pipeline_answer,
    render_comparison_table, render_badge, render_section_header,
    render_winner_banner, format_token_reduction
)

# ===== Page Config =====
st.set_page_config(
    page_title="GraphRAG Inference Benchmark",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===== Initialize Session State =====
if "query_results" not in st.session_state:
    st.session_state.query_results = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None
if "history" not in st.session_state:
    st.session_state.history = []


# ===== Sidebar =====
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("**LLM Model**")
    st.code(config.LLM_MODEL, language=None)
    
    st.markdown("**RAG Settings**")
    rag_top_k = st.slider("RAG Top-K Chunks", 1, 20, config.RAG_TOP_K, key="rag_topk")
    chunk_size = st.slider("Chunk Size (chars)", 200, 3000, config.CHUNK_SIZE, step=100, key="chunk_sz")
    
    st.markdown("**GraphRAG Settings**")
    gr_method = st.selectbox("Retrieval Method", ["hybrid", "community"], key="gr_method")
    gr_top_k = st.slider("GraphRAG Top-K", 1, 15, config.GRAPHRAG_TOP_K, key="gr_topk")
    gr_hops = st.slider("Num Hops", 1, 4, config.GRAPHRAG_NUM_HOPS, key="gr_hops")
    
    st.markdown("---")
    st.markdown("**Pipeline Status**")
    
    api_configured = bool(config.GROQ_API_KEY)
    tg_configured = bool(config.TG_HOST and config.TG_HOST != "http://localhost")
    
    st.markdown(f"{'✅' if api_configured else '❌'} Groq API Key")
    st.markdown(f"{'✅' if tg_configured else '⚠️'} TigerGraph Connected")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
        "Built for GraphRAG Inference Hackathon<br>by TigerGraph</div>",
        unsafe_allow_html=True
    )


# ===== Header =====
st.markdown(render_header(), unsafe_allow_html=True)


# ===== Main Tabs =====
tab_live, tab_benchmark, tab_results, tab_about = st.tabs([
    "🔴 Live Query", "📊 Full Benchmark", "📈 Past Results", "ℹ️ About"
])


# ===================================================================
# TAB 1: LIVE QUERY
# ===================================================================
with tab_live:
    st.markdown(render_section_header("🔍", "Query All Pipelines"), unsafe_allow_html=True)
    
    # Handle pending query from sample question buttons
    if "_pending_query" not in st.session_state:
        st.session_state._pending_query = ""
    
    default_query = st.session_state._pending_query if st.session_state._pending_query else ""
    
    # Query input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Enter your question",
            value=default_query,
            placeholder="e.g., Who is Geoffrey Hinton and what are his contributions to deep learning?",
            label_visibility="collapsed",
            key="live_query"
        )
    with col_btn:
        run_query = st.button("⚡ Run", use_container_width=True, key="run_btn")
    
    # Clear pending after it's been consumed
    if st.session_state._pending_query:
        st.session_state._pending_query = ""
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        for i, q in enumerate(config.BENCHMARK_QUESTIONS[:5]):
            if st.button(q["question"][:80] + "...", key=f"sample_{i}"):
                st.session_state._pending_query = q["question"]
                st.rerun()
    
    if run_query and query:
        st.markdown(render_section_header("⏳", "Running Pipelines..."), unsafe_allow_html=True)
        
        results = {}
        
        # Pipeline 1: LLM-Only
        with st.spinner("Running Pipeline 1: LLM-Only..."):
            from pipelines.llm_only import LLMOnlyPipeline
            p1 = LLMOnlyPipeline()
            p1.ensure_initialized()
            results["llm_only"] = p1.query(query)
        
        # Pipeline 2: Basic RAG
        with st.spinner("Running Pipeline 2: Basic RAG..."):
            from pipelines.basic_rag import BasicRAGPipeline
            p2 = BasicRAGPipeline()
            p2.ensure_initialized()
            results["basic_rag"] = p2.query(query)
        
        # Pipeline 3: GraphRAG
        with st.spinner("Running Pipeline 3: GraphRAG..."):
            try:
                from pipelines.graphrag_pipeline import GraphRAGPipeline
                p3 = GraphRAGPipeline()
                p3.ensure_initialized()
                results["graphrag"] = p3.query(query)
            except Exception as e:
                from pipelines.base import PipelineResult
                results["graphrag"] = PipelineResult(
                    pipeline_name="GraphRAG",
                    question=query,
                    answer=f"⚠️ GraphRAG not available: {e}\n\nConfigure TigerGraph in .env to enable.",
                    metadata={"error": str(e)}
                )
        
        st.session_state.query_results = results
        st.session_state.history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": {k: v.to_dict() for k, v in results.items()}
        })
    
    # ===== Display Results =====
    if st.session_state.query_results:
        results = st.session_state.query_results
        r1 = results["llm_only"]
        r2 = results["basic_rag"]
        r3 = results["graphrag"]
        
        # Top-level metrics
        st.markdown(render_section_header("📊", "Metrics Comparison"), unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        
        # Token comparison
        tokens = [r1.total_tokens, r2.total_tokens, r3.total_tokens]
        best_token = min(tokens)
        
        with m1:
            delta_text, delta_type = format_token_reduction(r2.total_tokens, r3.total_tokens)
            st.markdown(render_metric_card("Token Reduction vs RAG", delta_text, "GraphRAG advantage", delta_type), unsafe_allow_html=True)
        with m2:
            latencies = [r1.latency_seconds, r2.latency_seconds, r3.latency_seconds]
            fastest = ["LLM-Only", "Basic RAG", "GraphRAG"][latencies.index(min(latencies))]
            st.markdown(render_metric_card("Fastest Pipeline", fastest, f"{min(latencies):.2f}s", "positive"), unsafe_allow_html=True)
        with m3:
            st.markdown(render_metric_card("LLM-Only Tokens", f"{r1.total_tokens:,}", "No retrieval", "negative"), unsafe_allow_html=True)
        with m4:
            st.markdown(render_metric_card("GraphRAG Tokens", f"{r3.total_tokens:,}", "Graph-powered", "positive"), unsafe_allow_html=True)
        
        # Token Usage Chart
        st.markdown("")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            fig_tokens = go.Figure(data=[
                go.Bar(
                    x=["LLM-Only", "Basic RAG", "GraphRAG"],
                    y=[r1.total_tokens, r2.total_tokens, r3.total_tokens],
                    marker_color=["#f85149", "#58a6ff", "#3fb950"],
                    text=[f"{t:,}" for t in [r1.total_tokens, r2.total_tokens, r3.total_tokens]],
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=14, family="Inter")
                )
            ])
            fig_tokens.update_layout(
                title=dict(text="Total Tokens per Pipeline", font=dict(color="#e6edf3", size=16, family="Inter")),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,45,0.6)",
                font=dict(color="#8b949e", family="Inter"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Tokens"),
                height=380, margin=dict(t=50, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
        
        with chart_col2:
            fig_latency = go.Figure(data=[
                go.Bar(
                    x=["LLM-Only", "Basic RAG", "GraphRAG"],
                    y=[r1.latency_seconds, r2.latency_seconds, r3.latency_seconds],
                    marker_color=["#f85149", "#58a6ff", "#3fb950"],
                    text=[f"{l:.2f}s" for l in [r1.latency_seconds, r2.latency_seconds, r3.latency_seconds]],
                    textposition="outside",
                    textfont=dict(color="#e6edf3", size=14, family="Inter")
                )
            ])
            fig_latency.update_layout(
                title=dict(text="Response Latency", font=dict(color="#e6edf3", size=16, family="Inter")),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,45,0.6)",
                font=dict(color="#8b949e", family="Inter"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Seconds"),
                height=380, margin=dict(t=50, b=40, l=60, r=20),
            )
            st.plotly_chart(fig_latency, use_container_width=True)
        
        # Side-by-side responses
        st.markdown(render_section_header("💬", "Pipeline Responses"), unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(render_pipeline_answer(
                "🔴 Pipeline 1: LLM-Only", r1.answer,
                r1.total_tokens, r1.latency_seconds, r1.cost_usd, "pipeline-llm"
            ), unsafe_allow_html=True)
        with col2:
            st.markdown(render_pipeline_answer(
                "🔵 Pipeline 2: Basic RAG", r2.answer,
                r2.total_tokens, r2.latency_seconds, r2.cost_usd, "pipeline-rag"
            ), unsafe_allow_html=True)
        with col3:
            st.markdown(render_pipeline_answer(
                "🟢 Pipeline 3: GraphRAG", r3.answer,
                r3.total_tokens, r3.latency_seconds, r3.cost_usd, "pipeline-graphrag"
            ), unsafe_allow_html=True)
        
        # Detailed metrics table
        with st.expander("📋 Detailed Metrics Table"):
            table_data = [
                {
                    "Metric": "Prompt Tokens",
                    "LLM-Only": f"{r1.prompt_tokens:,}",
                    "Basic RAG": f"{r2.prompt_tokens:,}",
                    "GraphRAG": f"{r3.prompt_tokens:,}",
                },
                {
                    "Metric": "Completion Tokens",
                    "LLM-Only": f"{r1.completion_tokens:,}",
                    "Basic RAG": f"{r2.completion_tokens:,}",
                    "GraphRAG": f"{r3.completion_tokens:,}",
                },
                {
                    "Metric": "Total Tokens",
                    "LLM-Only": f"{r1.total_tokens:,}",
                    "Basic RAG": f"{r2.total_tokens:,}",
                    "GraphRAG": f"{r3.total_tokens:,}",
                },
                {
                    "Metric": "Latency (s)",
                    "LLM-Only": f"{r1.latency_seconds:.3f}",
                    "Basic RAG": f"{r2.latency_seconds:.3f}",
                    "GraphRAG": f"{r3.latency_seconds:.3f}",
                },
                {
                    "Metric": "Cost (USD)",
                    "LLM-Only": f"${r1.cost_usd:.6f}",
                    "Basic RAG": f"${r2.cost_usd:.6f}",
                    "GraphRAG": f"${r3.cost_usd:.6f}",
                },
                {
                    "Metric": "Context Tokens",
                    "LLM-Only": "0 (none)",
                    "Basic RAG": f"{r2.context_tokens:,}",
                    "GraphRAG": f"{r3.context_tokens:,}",
                },
            ]
            st.markdown(render_comparison_table(table_data), unsafe_allow_html=True)


# ===================================================================
# TAB 2: FULL BENCHMARK
# ===================================================================
with tab_benchmark:
    st.markdown(render_section_header("🏋️", "Run Full Benchmark Suite"), unsafe_allow_html=True)
    
    st.markdown("""
    Run all **10 benchmark questions** through all 3 pipelines, then evaluate accuracy 
    with **LLM-as-a-Judge** and **BERTScore**. This generates the complete comparison report.
    """)
    
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        num_questions = st.slider("Number of questions", 1, len(config.BENCHMARK_QUESTIONS), len(config.BENCHMARK_QUESTIONS), key="num_q")
    with bcol2:
        skip_graphrag = st.checkbox("Skip GraphRAG (test mode)", value=False, key="skip_gr")
    
    if st.button("🚀 Run Full Benchmark", use_container_width=True, key="run_benchmark"):
        with st.spinner("Running full benchmark... This may take a few minutes."):
            from evaluation.benchmark import run_full_benchmark
            questions = config.BENCHMARK_QUESTIONS[:num_questions]
            benchmark = run_full_benchmark(questions=questions, skip_graphrag=skip_graphrag)
            st.session_state.benchmark_results = benchmark
        st.success("✅ Benchmark complete!")
    
    # Display benchmark results
    if st.session_state.benchmark_results:
        bm = st.session_state.benchmark_results
        
        # Winner banner
        if not bm["metadata"].get("skip_graphrag") and "graphrag" in bm:
            reduction = bm["comparison"].get("token_reduction_vs_rag_pct", 0)
            pass_rate = bm["judge"].get("graphrag", {}).get("pass_rate", 0)
            if reduction > 0:
                st.markdown(render_winner_banner("GraphRAG", reduction, pass_rate), unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown(render_section_header("📊", "Benchmark Summary"), unsafe_allow_html=True)
        
        pipelines_list = ["llm_only", "basic_rag"]
        if not bm["metadata"].get("skip_graphrag"):
            pipelines_list.append("graphrag")
        
        summary_rows = []
        for pkey in pipelines_list:
            pdata = bm.get(pkey, {})
            jdata = bm.get("judge", {}).get(pkey, {})
            bsdata = bm.get("bertscore", {}).get(pkey, {})
            
            pr = jdata.get("pass_rate", 0)
            pr_badge = render_badge(f"{pr:.0%} PASS", "pass" if pr >= 0.7 else "fail")
            
            summary_rows.append({
                "Pipeline": pdata.get("pipeline", pkey),
                "Avg Tokens": f"{pdata.get('avg_tokens', 0):,.0f}",
                "Avg Latency": f"{pdata.get('avg_latency', 0):.3f}s",
                "Avg Cost": f"${pdata.get('avg_cost', 0):.6f}",
                "Pass Rate": f"{pr:.0%}",
                "BERTScore F1": f"{bsdata.get('avg_f1_rescaled', 0):.4f}",
            })
        
        st.markdown(render_comparison_table(summary_rows), unsafe_allow_html=True)
        
        # Charts
        st.markdown("")
        ch1, ch2, ch3 = st.columns(3)
        
        names = [bm[pk]["pipeline"] for pk in pipelines_list]
        colors = ["#f85149", "#58a6ff", "#3fb950"][:len(pipelines_list)]
        
        with ch1:
            fig = go.Figure(data=[go.Bar(
                x=names, y=[bm[pk]["avg_tokens"] for pk in pipelines_list],
                marker_color=colors,
                text=[f"{bm[pk]['avg_tokens']:,.0f}" for pk in pipelines_list],
                textposition="outside", textfont=dict(color="#e6edf3", size=13)
            )])
            fig.update_layout(
                title=dict(text="Avg Tokens", font=dict(color="#e6edf3", size=14)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,45,0.6)",
                font=dict(color="#8b949e"), height=350, margin=dict(t=50, b=30),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with ch2:
            judge_rates = [bm["judge"].get(pk, {}).get("pass_rate", 0) for pk in pipelines_list]
            fig = go.Figure(data=[go.Bar(
                x=names, y=[r * 100 for r in judge_rates],
                marker_color=colors,
                text=[f"{r:.0%}" for r in judge_rates],
                textposition="outside", textfont=dict(color="#e6edf3", size=13)
            )])
            fig.update_layout(
                title=dict(text="LLM Judge Pass Rate", font=dict(color="#e6edf3", size=14)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,45,0.6)",
                font=dict(color="#8b949e"), height=350, margin=dict(t=50, b=30),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="%", range=[0, 110])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with ch3:
            bert_scores = [bm["bertscore"].get(pk, {}).get("avg_f1_rescaled", 0) for pk in pipelines_list]
            fig = go.Figure(data=[go.Bar(
                x=names, y=bert_scores,
                marker_color=colors,
                text=[f"{s:.4f}" for s in bert_scores],
                textposition="outside", textfont=dict(color="#e6edf3", size=13)
            )])
            fig.update_layout(
                title=dict(text="BERTScore F1 (rescaled)", font=dict(color="#e6edf3", size=14)),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(22,27,45,0.6)",
                font=dict(color="#8b949e"), height=350, margin=dict(t=50, b=30),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Per-question breakdown
        with st.expander("📋 Per-Question LLM Judge Results"):
            for pkey in pipelines_list:
                jdata = bm.get("judge", {}).get(pkey, {})
                pname = bm[pkey]["pipeline"]
                st.markdown(f"**{pname}** — Pass Rate: {jdata.get('pass_rate', 0):.0%}")
                for j in jdata.get("judgments", []):
                    icon = "✅" if j["verdict"] == "PASS" else "❌"
                    st.markdown(f"  {icon} Q{j['question_index']+1}: {j['question'][:70]}... → **{j['verdict']}**")
                st.markdown("---")
        
        # Export
        if st.button("💾 Export Results as JSON", key="export_bm"):
            filepath = os.path.join(config.RESULTS_DIR, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(filepath, "w") as f:
                json.dump(bm, f, indent=2, default=str)
            st.success(f"Saved to {filepath}")


# ===================================================================
# TAB 3: PAST RESULTS
# ===================================================================
with tab_results:
    st.markdown(render_section_header("📈", "Past Benchmark Results"), unsafe_allow_html=True)
    
    result_files = []
    if os.path.exists(config.RESULTS_DIR):
        result_files = sorted(
            [f for f in os.listdir(config.RESULTS_DIR) if f.endswith(".json")],
            reverse=True
        )
    
    if result_files:
        selected_file = st.selectbox("Select a result file", result_files, key="past_result")
        filepath = os.path.join(config.RESULTS_DIR, selected_file)
        
        with open(filepath, "r") as f:
            past_data = json.load(f)
        
        st.json(past_data.get("comparison", {}))
        st.json(past_data.get("metadata", {}))
    else:
        st.info("No past results found. Run a full benchmark first!")
    
    # Query history from this session
    if st.session_state.history:
        st.markdown(render_section_header("📜", "Session Query History"), unsafe_allow_html=True)
        for i, entry in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Query {len(st.session_state.history)-i}: {entry['query'][:60]}..."):
                st.markdown(f"**Time:** {entry['timestamp']}")
                for pname, pdata in entry["results"].items():
                    st.markdown(f"**{pdata['pipeline_name']}**: {pdata['total_tokens']} tokens, {pdata['latency_seconds']}s")
                    st.markdown(f"> {pdata['answer'][:200]}...")


# ===================================================================
# TAB 4: ABOUT
# ===================================================================
with tab_about:
    st.markdown(render_section_header("ℹ️", "About This Project"), unsafe_allow_html=True)
    
    st.markdown("""
    ### GraphRAG Inference Benchmark Dashboard
    
    This project was built for the **GraphRAG Inference Hackathon by TigerGraph**. 
    It proves that **graphs make LLM inference faster, cheaper, and smarter** than 
    vector-based RAG alone.
    
    #### Architecture
    
    | Pipeline | Description | Technology |
    |----------|-------------|------------|
    | **Pipeline 1: LLM-Only** | Direct LLM query, no retrieval | Groq Llama 3.3 70B |
    | **Pipeline 2: Basic RAG** | Vector similarity search + LLM | ChromaDB + Groq |
    | **Pipeline 3: GraphRAG** | Knowledge graph + multi-hop reasoning + LLM | TigerGraph + Groq |
    
    #### Evaluation Methods
    
    - **LLM-as-a-Judge**: Llama 3.3 70B grades each answer PASS/FAIL
    - **BERTScore**: Measures semantic similarity (DeBERTa model)
    
    #### Key Metrics
    
    - **Token Reduction**: % improvement in tokens vs Basic RAG
    - **Answer Accuracy**: Quality maintained or improved
    - **Response Latency**: End-to-end time
    - **Cost per Query**: Calculated from token usage
    
    #### Tech Stack
    
    - **LLM**: Groq Llama 3.3 70B (free tier, blazing fast)
    - **Embeddings**: Sentence-Transformers all-MiniLM-L6-v2 (local)
    - **Vector DB**: ChromaDB (local)
    - **Graph DB**: TigerGraph Savanna
    - **GraphRAG**: TigerGraph GraphRAG repo
    - **Dashboard**: Streamlit + Plotly
    - **Dataset**: Wikipedia AI/ML articles (2M+ tokens)
    - **Evaluation**: HuggingFace BERTScore
    
    ---
    
    *Built with ❤️ for the GraphRAG Inference Hackathon by TigerGraph*
    """)
