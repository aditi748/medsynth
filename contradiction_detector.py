from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
import numpy as np

# We leave the global definitions for constants
SIMILARITY_THRESHOLD = 0.75  
CONTRADICTION_TYPES  = ["OUTCOME", "POPULATION", "DOSAGE", "METHODOLOGY"]

def get_embeddings(papers, embedding_model): # Added model as an argument
    summaries = []
    for p in papers:
        text = p.get("summary", "")
        if not text or "No abstract" in text:
            text = p.get("title", "")
        summaries.append(text)
    return embedding_model.encode(summaries)

def find_similar_pairs(papers, embeddings):
    pairs  = []
    matrix = cosine_similarity(embeddings)
    n      = len(papers)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] >= SIMILARITY_THRESHOLD:
                pairs.append({
                    "paper_a_index": i,
                    "paper_b_index": j,
                    "similarity"   : round(float(matrix[i][j]), 3)
                })
    return pairs

def check_contradiction(paper_a, paper_b):
    # Initialize Groq inside or ensure it's global - using your existing setup
    groq_client = Groq(api_key=GROQ_API_KEY)
    facts_a = paper_a.get("facts", {})
    facts_b = paper_b.get("facts", {})

    prompt = f"""You are a senior medical research analyst.
Compare these two papers and determine if they contradict each other.
... (rest of your existing prompt) ...
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1
    )

    raw    = response.choices[0].message.content.strip()
    result = {
        "is_contradiction": False,
        "type"            : None,
        "finding_a"       : None,
        "finding_b"       : None,
        "reason"          : None,
        "clinical"        : None,
    }

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("CONTRADICTION:"):
            result["is_contradiction"] = "YES" in line.upper()
        elif line.startswith("TYPE:"):
            t = line.replace("TYPE:", "").strip()
            if t in CONTRADICTION_TYPES:
                result["type"] = t
        elif line.startswith("FINDING_A:"):
            result["finding_a"] = line.replace("FINDING_A:", "").strip()
        elif line.startswith("FINDING_B:"):
            result["finding_b"] = line.replace("FINDING_B:", "").strip()
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("CLINICAL:"):
            result["clinical"] = line.replace("CLINICAL:", "").strip()

    return result

def detect_contradictions(papers):
    # CRITICAL: Initialize model HERE so it uses the HF_TOKEN from main.py
    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"\nRunning contradiction detection on {len(papers)} papers...")

    if len(papers) < 2:
        print("  Need at least 2 papers.")
        return []

    print("  Computing embeddings...")
    embeddings = get_embeddings(papers, embedding_model)

    print("  Finding similar paper pairs...")
    pairs = find_similar_pairs(papers, embeddings)
    print(f"  Found {len(pairs)} similar pairs above threshold {SIMILARITY_THRESHOLD}")

    if not pairs:
        print("  No similar pairs found — no contradictions possible.")
        return []

    contradictions = []
    for pair in pairs:
        paper_a = papers[pair["paper_a_index"]]
        paper_b = papers[pair["paper_b_index"]]
        print(f"  Checking: '{paper_a['title'][:40]}...' vs '{paper_b['title'][:40]}...'")

        result = check_contradiction(paper_a, paper_b)

        if result["is_contradiction"]:
            contradictions.append({
                "paper_a_title"  : paper_a["title"],
                "paper_b_title"  : paper_b["title"],
                "paper_a_year"   : paper_a["year"],
                "paper_b_year"   : paper_b["year"],
                "paper_a_journal": paper_a.get("journal", ""),
                "paper_a_score"  : paper_a.get("evidence_score", 0),
                "paper_b_score"  : paper_b.get("evidence_score", 0),
                "paper_a_sample" : paper_a.get("sample_size", 0),
                "paper_b_sample" : paper_b.get("sample_size", 0),
                "similarity"     : pair["similarity"],
                "type"           : result["type"],
                "finding_a"      : result["finding_a"],
                "finding_b"      : result["finding_b"],
                "reason"         : result["reason"],
                "clinical"       : result["clinical"],
            })

    return contradictions