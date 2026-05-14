"""
GraphRAG Inference Hackathon — Configuration Module
====================================================
Centralized configuration for all pipelines, evaluation, and dashboard.
Loads from .env file and provides typed access to all settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Windows Unicode Fix ---
import builtins
_original_print = builtins.print
def _safe_print(*args, **kwargs):
    try:
        _original_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode('ascii', 'ignore').decode('ascii') for a in args]
        _original_print(*safe_args, **kwargs)
builtins.print = _safe_print
# ---------------------------


# =============================================================================
# LLM Configuration (Groq — blazing fast free inference)
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")   # Optional: HuggingFace token for higher rate limits
LLM_MODEL = "llama-3.3-70b-versatile"         # Best quality on Groq, 128K context
LLM_FAST_MODEL = "llama-3.1-8b-instant"       # Fast model for judging (saves rate limits)

# Groq pricing — free tier
# 14,400 requests/day, 6,000 tokens/min for 70b
LLM_INPUT_COST_PER_1M = 0.59    # $0.59 per 1M input tokens
LLM_OUTPUT_COST_PER_1M = 0.79   # $0.79 per 1M output tokens
USE_FREE_TIER = True             # Set True to calculate cost as $0 (free tier)

# =============================================================================
# BERTScore Configuration
# =============================================================================
BERTSCORE_MODEL = os.getenv("BERTSCORE_MODEL", "roberta-large")


# =============================================================================
# Embedding Configuration (Local — sentence-transformers, free & fast)
# =============================================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, free, no API needed
EMBEDDING_DIMENSION = 384


# =============================================================================
# TigerGraph Configuration
# =============================================================================
TG_HOST = os.getenv("TG_HOST", "http://localhost")
TG_USERNAME = os.getenv("TG_USERNAME", "tigergraph")
TG_PASSWORD = os.getenv("TG_PASSWORD", "tigergraph")
TG_SECRET = os.getenv("TG_SECRET", "")
TG_GRAPHNAME = os.getenv("TG_GRAPHNAME", "GraphRAGHackathon")
TG_RESTPP_PORT = os.getenv("TG_RESTPP_PORT", "14240")
TG_GS_PORT = os.getenv("TG_GS_PORT", "14240")
GRAPHRAG_HOST = os.getenv("GRAPHRAG_HOST", "http://localhost:8000")


# =============================================================================
# ChromaDB Configuration (Pipeline 2: Basic RAG)
# =============================================================================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "graphrag_hackathon"

# RAG Chunking Parameters
CHUNK_SIZE = 1000           # Characters per chunk
CHUNK_OVERLAP = 200         # Overlap between chunks
RAG_TOP_K = 5               # Number of chunks to retrieve


# =============================================================================
# GraphRAG Retrieval Parameters (Pipeline 3)
# =============================================================================
GRAPHRAG_METHOD = "hybrid"   # "hybrid" or "community"
GRAPHRAG_TOP_K = 5           # Seed results per search
GRAPHRAG_NUM_HOPS = 2        # Graph traversal depth
GRAPHRAG_NUM_SEEN_MIN = 2    # Minimum occurrence filter
GRAPHRAG_COMMUNITY_LEVEL = 2 # Community hierarchy level


# =============================================================================
# Dataset Configuration
# =============================================================================
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset", "data")
DATASET_DOMAIN = "Artificial Intelligence"  # Wikipedia category
MIN_TOKENS = 2_000_000       # Minimum 2M tokens required


# =============================================================================
# Evaluation Configuration
# =============================================================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"  # BERTScore model


# =============================================================================
# Benchmark Questions (used for evaluation across all 3 pipelines)
# =============================================================================
BENCHMARK_QUESTIONS = [
    {
        "question": "Who is considered the father of artificial intelligence and what were his key contributions?",
        "expected_answer": "John McCarthy is widely considered the father of artificial intelligence. He coined the term 'artificial intelligence' in 1956 at the Dartmouth Conference, developed the Lisp programming language, and made fundamental contributions to the field including the concept of time-sharing in computing."
    },
    {
        "question": "How does the transformer architecture work and why was it revolutionary for natural language processing?",
        "expected_answer": "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al., uses self-attention mechanisms to process all positions in a sequence simultaneously rather than sequentially. This parallel processing was revolutionary because it enabled much faster training, better handling of long-range dependencies, and led to breakthroughs like BERT, GPT, and modern large language models."
    },
    {
        "question": "What is the relationship between Geoffrey Hinton, Yann LeCun, and Yoshua Bengio in the development of deep learning?",
        "expected_answer": "Geoffrey Hinton, Yann LeCun, and Yoshua Bengio are known as the 'Godfathers of Deep Learning' and jointly received the 2018 Turing Award. Hinton pioneered backpropagation and deep belief networks, LeCun developed convolutional neural networks (LeNet), and Bengio advanced recurrent neural networks and generative adversarial networks. Their collective work laid the foundation for modern deep learning."
    },
    {
        "question": "What are the differences between supervised learning, unsupervised learning, and reinforcement learning?",
        "expected_answer": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs (e.g., classification, regression). Unsupervised learning finds patterns in unlabeled data (e.g., clustering, dimensionality reduction). Reinforcement learning learns through trial and error by interacting with an environment and receiving rewards or penalties for actions taken."
    },
    {
        "question": "How did AlphaGo defeat Lee Sedol and what was the significance of this achievement?",
        "expected_answer": "AlphaGo, developed by DeepMind, defeated world champion Go player Lee Sedol 4-1 in March 2016. It used a combination of deep neural networks and Monte Carlo tree search, trained on millions of human games and through self-play. This was significant because Go has more possible positions than atoms in the universe, making it a milestone in AI's ability to handle complex strategic reasoning."
    },
    {
        "question": "What is the history and evolution of natural language processing from rule-based systems to modern LLMs?",
        "expected_answer": "NLP evolved from rule-based systems in the 1950s-1980s (like ELIZA and SHRDLU) to statistical methods in the 1990s-2000s (using probabilistic models and machine learning). The introduction of word embeddings (Word2Vec, 2013) and then the transformer architecture (2017) led to the current era of large language models like GPT, BERT, and their successors, which achieve human-level or better performance on many language tasks."
    },
    {
        "question": "What organizations and companies have been most influential in AI research, and how are they connected?",
        "expected_answer": "Key organizations include DeepMind (acquired by Google/Alphabet), OpenAI (initially nonprofit, backed by Microsoft), Meta AI (formerly Facebook AI Research), Stanford AI Lab, MIT CSAIL, and Carnegie Mellon. Many researchers have moved between these institutions. Google Brain merged with DeepMind. OpenAI's founding team included researchers from Google and Stanford. These organizations are connected through collaborative research, competition, and the flow of talent."
    },
    {
        "question": "What ethical concerns have been raised about artificial intelligence and how are they being addressed?",
        "expected_answer": "Key ethical concerns include algorithmic bias and fairness, privacy and surveillance, job displacement, autonomous weapons, AI safety and alignment, deepfakes and misinformation. These are being addressed through AI ethics guidelines (EU AI Act, UNESCO recommendations), responsible AI frameworks by companies, research into explainable AI (XAI), bias detection tools, and organizations like Partnership on AI and the AI Safety Institute."
    },
    {
        "question": "How do convolutional neural networks work and what are their main applications in computer vision?",
        "expected_answer": "CNNs use convolutional layers with learnable filters that scan across input images to detect features like edges, textures, and shapes. They typically include pooling layers for dimensionality reduction and fully connected layers for classification. Key applications include image classification (ImageNet), object detection (YOLO, R-CNN), facial recognition, medical image analysis, and autonomous driving."
    },
    {
        "question": "What is the connection between the Turing Test, the Chinese Room argument, and the current debate about AI consciousness?",
        "expected_answer": "The Turing Test (1950) proposed that a machine is intelligent if it can fool a human into thinking it's human through conversation. John Searle's Chinese Room argument (1980) challenged this by arguing that a system can manipulate symbols without understanding their meaning, suggesting the Turing Test is insufficient for measuring true intelligence. These debates continue today with modern LLMs that pass variants of the Turing Test, raising questions about whether statistical pattern matching constitutes understanding or consciousness."
    },
]


# =============================================================================
# Dashboard Configuration
# =============================================================================
DASHBOARD_TITLE = "GraphRAG Inference Benchmark"
DASHBOARD_SUBTITLE = "Proving Graph Beats Tokens"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
