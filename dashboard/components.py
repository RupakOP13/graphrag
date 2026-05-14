"""
Dashboard Components — Reusable UI Elements
=============================================
HTML/Markdown generators for the Streamlit dashboard.
"""


def render_header():
    """Render the main dashboard header."""
    return """
    <div class="main-header">
        <h1>⚡ GraphRAG Inference Benchmark</h1>
        <div class="subtitle">Proving Graph Beats Tokens — Side-by-Side Pipeline Comparison</div>
    </div>
    """


def render_metric_card(label, value, delta=None, delta_type="neutral"):
    """Render a single metric card with optional delta."""
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def render_pipeline_answer(pipeline_name, answer, tokens, latency, cost, accent_class="pipeline-llm"):
    """Render a pipeline response card."""
    return f"""
    <div class="pipeline-card {accent_class}">
        <h3>{pipeline_name}</h3>
        <div style="display: flex; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <div><span style="color: #8b949e; font-size: 0.8rem;">TOKENS</span><br>
                <span style="font-weight: 700; font-size: 1.1rem; color: #e6edf3;">{tokens:,}</span></div>
            <div><span style="color: #8b949e; font-size: 0.8rem;">LATENCY</span><br>
                <span style="font-weight: 700; font-size: 1.1rem; color: #e6edf3;">{latency:.2f}s</span></div>
            <div><span style="color: #8b949e; font-size: 0.8rem;">COST</span><br>
                <span style="font-weight: 700; font-size: 1.1rem; color: #e6edf3;">${cost:.6f}</span></div>
        </div>
        <div class="answer-text">{answer}</div>
    </div>
    """


def render_comparison_table(data_rows):
    """
    Render a comparison table.
    data_rows: list of dicts with keys matching headers
    """
    if not data_rows:
        return "<p>No data available</p>"
    
    headers = list(data_rows[0].keys())
    
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    
    rows_html = ""
    for row in data_rows:
        cells = "".join(f"<td>{v}</td>" for v in row.values())
        rows_html += f"<tr>{cells}</tr>"
    
    return f"""
    <table class="comparison-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """


def render_badge(text, badge_type="pass"):
    """Render a badge (pass/fail/winner)."""
    return f'<span class="badge badge-{badge_type}">{text}</span>'


def render_section_header(icon, text):
    """Render a section header with icon."""
    return f'<div class="section-header">{icon} {text}</div>'


def render_winner_banner(winner_name, token_reduction_pct, pass_rate):
    """Render a winner announcement banner."""
    return f"""
    <div style="
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🏆</div>
        <div style="font-size: 1.4rem; font-weight: 800; color: #a78bfa; margin-bottom: 0.5rem;">
            {winner_name} Wins!
        </div>
        <div style="color: #8b949e; font-size: 0.95rem;">
            <strong style="color: #3fb950;">{token_reduction_pct:.1f}%</strong> token reduction with 
            <strong style="color: #3fb950;">{pass_rate:.0%}</strong> accuracy pass rate
        </div>
    </div>
    """


def format_token_reduction(baseline_tokens, comparison_tokens):
    """Calculate and format token reduction percentage."""
    if baseline_tokens == 0:
        return "N/A", "neutral"
    reduction = ((baseline_tokens - comparison_tokens) / baseline_tokens) * 100
    if reduction > 0:
        return f"↓ {reduction:.1f}%", "positive"
    elif reduction < 0:
        return f"↑ {abs(reduction):.1f}%", "negative"
    else:
        return "0%", "neutral"
