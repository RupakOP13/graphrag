"""
Dashboard Styles — Premium Dark Theme
=======================================
Custom CSS for the Streamlit dashboard to create a stunning, judge-worthy interface.
"""

CUSTOM_CSS = """
<style>
    /* ===== GLOBAL ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0f1e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== HEADER ===== */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .main-header .subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        font-weight: 400;
        letter-spacing: 0.05em;
    }
    
    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: linear-gradient(145deg, rgba(22, 27, 45, 0.9), rgba(15, 18, 35, 0.95));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #e6edf3;
        line-height: 1.2;
    }
    .metric-card .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    .metric-delta.positive { color: #3fb950; }
    .metric-delta.negative { color: #f85149; }
    .metric-delta.neutral { color: #8b949e; }
    
    /* ===== PIPELINE CARDS ===== */
    .pipeline-card {
        background: rgba(22, 27, 45, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .pipeline-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
    }
    .pipeline-card h3 {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .pipeline-card .answer-text {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.9rem;
        color: #c9d1d9;
        line-height: 1.6;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Pipeline-specific accent colors */
    .pipeline-llm { border-left: 4px solid #f85149; }
    .pipeline-rag { border-left: 4px solid #58a6ff; }
    .pipeline-graphrag { border-left: 4px solid #3fb950; }
    
    /* ===== BADGE ===== */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-pass { background: rgba(63, 185, 80, 0.15); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }
    .badge-fail { background: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.3); }
    .badge-winner { background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2)); color: #a78bfa; border: 1px solid rgba(168, 85, 247, 0.4); }
    
    /* ===== COMPARISON TABLE ===== */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        background: rgba(22, 27, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .comparison-table th {
        background: rgba(99, 102, 241, 0.1);
        color: #8b949e;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        padding: 1rem;
        text-align: left;
    }
    .comparison-table td {
        padding: 0.85rem 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.04);
        color: #c9d1d9;
        font-size: 0.9rem;
    }
    .comparison-table tr:hover td {
        background: rgba(99, 102, 241, 0.05);
    }
    .highlight-best {
        color: #3fb950 !important;
        font-weight: 700;
    }
    
    /* ===== SECTION DIVIDERS ===== */
    .section-header {
        font-size: 1.4rem;
        font-weight: 800;
        color: #e6edf3;
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(99, 102, 241, 0.2);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.5); }
    
    /* ===== STREAMLIT OVERRIDES ===== */
    .stTextInput > div > div > input {
        background: rgba(22, 27, 45, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.35) !important;
    }
    .stSelectbox > div > div {
        background: rgba(22, 27, 45, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(22, 27, 45, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        color: #8b949e;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #e6edf3 !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(22, 27, 45, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
    }
    .stSpinner > div > div {
        border-top-color: #667eea !important;
    }
</style>
"""
