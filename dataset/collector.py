"""
Dataset Collector — Wikipedia AI/ML Articles
=============================================
Downloads Wikipedia articles related to Artificial Intelligence and Machine Learning.
Targets ≥2M tokens for the hackathon requirement.
"""

import os
import json
import time
import wikipediaapi
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Core seed articles — these have rich entity connections
SEED_ARTICLES = [
    # Core AI topics
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Neural network", "Natural language processing", "Computer vision",
    "Reinforcement learning", "Supervised learning", "Unsupervised learning",
    
    # Architectures & Models
    "Transformer (deep learning architecture)", "Convolutional neural network",
    "Recurrent neural network", "Generative adversarial network",
    "Long short-term memory", "Autoencoder", "Diffusion model",
    "Variational autoencoder", "Boltzmann machine",
    
    # Key Models & Systems
    "GPT-4", "GPT-3", "BERT (language model)", "ChatGPT",
    "AlphaGo", "AlphaFold", "DALL-E", "Stable Diffusion",
    "LLaMA (language model)", "Claude (language model)",
    "Midjourney", "GitHub Copilot",
    
    # Key People
    "Geoffrey Hinton", "Yann LeCun", "Yoshua Bengio",
    "Andrew Ng", "Fei-Fei Li", "Demis Hassabis",
    "Ian Goodfellow", "Andrej Karpathy", "Ilya Sutskever",
    "Sam Altman", "John McCarthy (computer scientist)",
    "Alan Turing", "Marvin Minsky", "Herbert A. Simon",
    
    # Organizations
    "OpenAI", "DeepMind", "Google Brain", "Meta AI",
    "Stanford Artificial Intelligence Laboratory",
    "MIT Computer Science and Artificial Intelligence Laboratory",
    "NVIDIA", "Anthropic",
    
    # Concepts & Methods
    "Backpropagation", "Gradient descent", "Attention (machine learning)",
    "Word embedding", "Word2Vec", "Transfer learning",
    "Few-shot learning", "Zero-shot learning", "Fine-tuning (deep learning)",
    "Prompt engineering", "Retrieval-augmented generation",
    "Knowledge graph", "Graph neural network",
    
    # Applications
    "Autonomous car", "Speech recognition", "Machine translation",
    "Facial recognition system", "Recommendation system",
    "Medical imaging", "Drug discovery",
    
    # Ethics & Safety
    "AI safety", "Artificial general intelligence",
    "Existential risk from AI", "AI alignment",
    "Algorithmic bias", "Explainable artificial intelligence",
    "Deepfake", "AI-generated art",
    
    # Benchmarks & Datasets
    "ImageNet", "MNIST database", "Common Crawl",
    "Turing test", "Chinese room",
    
    # History
    "History of artificial intelligence",
    "Dartmouth workshop", "AI winter",
    "Expert system", "Logic programming",
    
    # Mathematics & Foundations
    "Bayesian network", "Support vector machine",
    "Decision tree", "Random forest",
    "K-nearest neighbors algorithm", "Principal component analysis",
    "Dimensionality reduction", "Feature engineering",
    
    # Recent Developments
    "Foundation model", "Mixture of experts",
    "Multimodal learning", "Self-supervised learning",
    "Contrastive learning", "Neural architecture search",
    "Federated learning", "Edge computing",
    
    # Additional rich topics
    "Robotics", "Swarm intelligence", "Evolutionary computation",
    "Genetic algorithm", "Fuzzy logic",
    "Semantic Web", "Ontology (information science)",
    "Information retrieval", "Text mining",
    "Sentiment analysis", "Named-entity recognition",
    "Question answering", "Dialogue system",
    "Optical character recognition",
]


def count_tokens_approx(text: str) -> int:
    """Approximate token count (1 token ≈ 4 characters for English text)."""
    return len(text) // 4


def collect_articles(target_tokens: int = config.MIN_TOKENS) -> dict:
    """
    Collect Wikipedia articles until we reach the target token count.
    
    Returns:
        dict with 'articles' list and metadata
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="GraphRAGHackathon/1.0 (graphrag-inference-benchmark)",
        language="en"
    )
    
    articles = []
    total_tokens = 0
    seen_titles = set()
    queue = list(SEED_ARTICLES)
    
    print(f"🎯 Target: {target_tokens:,} tokens")
    print(f"📚 Starting with {len(queue)} seed articles\n")
    
    pbar = tqdm(total=target_tokens, unit="tok", desc="Collecting tokens")
    
    while total_tokens < target_tokens and queue:
        title = queue.pop(0)
        
        if title in seen_titles:
            continue
        seen_titles.add(title)
        
        try:
            page = wiki.page(title)
            if not page.exists():
                continue
            
            text = page.text
            if len(text) < 500:  # Skip stub articles
                continue
            
            tokens = count_tokens_approx(text)
            
            article = {
                "title": page.title,
                "url": page.fullurl,
                "text": text,
                "summary": page.summary[:500],
                "tokens": tokens,
                "categories": [cat for cat in list(page.categories.keys())[:10]],
            }
            
            articles.append(article)
            total_tokens += tokens
            pbar.update(tokens)
            
            # Add linked articles to queue for more coverage
            if total_tokens < target_tokens:
                links = list(page.links.keys())[:20]  # Top 20 links
                for link in links:
                    if link not in seen_titles and not link.startswith(("Wikipedia:", "Help:", "Template:", "Category:", "Portal:", "File:", "Draft:", "Talk:")):
                        queue.append(link)
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"\n⚠️  Error fetching '{title}': {e}")
            continue
    
    pbar.close()
    
    result = {
        "domain": config.DATASET_DOMAIN,
        "total_articles": len(articles),
        "total_tokens": total_tokens,
        "total_characters": sum(len(a["text"]) for a in articles),
        "articles": articles,
    }
    
    print(f"\n✅ Collected {len(articles)} articles")
    print(f"📊 Total tokens: {total_tokens:,}")
    print(f"📊 Total characters: {result['total_characters']:,}")
    
    return result


def save_dataset(data: dict, filename: str = "wikipedia_ai_dataset.json"):
    """Save the collected dataset to a JSON file."""
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    filepath = os.path.join(config.DATASET_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 Dataset saved to {filepath} ({size_mb:.1f} MB)")
    return filepath


def save_as_jsonl(data: dict, filename: str = "wikipedia_ai_dataset.jsonl"):
    """Save articles in JSONL format for TigerGraph ingestion."""
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    filepath = os.path.join(config.DATASET_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for article in data["articles"]:
            entry = {
                "title": article["title"],
                "text": article["text"],
                "source": article["url"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"💾 JSONL saved to {filepath} ({size_mb:.1f} MB)")
    return filepath


def load_dataset(filename: str = "wikipedia_ai_dataset.json") -> dict:
    """Load a previously saved dataset."""
    filepath = os.path.join(config.DATASET_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📂 Loaded {data['total_articles']} articles ({data['total_tokens']:,} tokens)")
    return data


if __name__ == "__main__":
    print("=" * 60)
    print("📖 Wikipedia AI/ML Dataset Collector")
    print("=" * 60)
    
    data = collect_articles()
    save_dataset(data)
    save_as_jsonl(data)
    
    print("\n" + "=" * 60)
    print("✅ Dataset collection complete!")
    print(f"   Articles: {data['total_articles']}")
    print(f"   Tokens:   {data['total_tokens']:,}")
    print("=" * 60)
