from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

SIMILARITY_THRESHOLD = 0.70  # lowered from 0.75 — more pairs get checked
CONTRADICTION_TYPES  = ["OUTCOME", "POPULATION", "DOSAGE", "METHODOLOGY"]

print("Loading sentence transformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client     = Groq(api_key=GROQ_API_KEY)


def get_embeddings(papers):
    texts = []
    for p in papers:
        # Use abstract for richer semantic comparison
        # Fall back to summary then title if abstract missing
        abstract = p.get("abstract", "")
        summary  = p.get("summary", "")
        title    = p.get("title", "")

        if abstract and len(abstract) > 100:
            text = abstract[:1000]
        elif summary and len(summary) > 50:
            text = summary
        else:
            text = title

        texts.append(text)

    return embedding_model.encode(texts)


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
    facts_a = paper_a.get("facts", {})
    facts_b = paper_b.get("facts", {})

    # Use conclusion from facts if available, otherwise use summary
    conclusion_a = facts_a.get("conclusion", "") or paper_a.get("summary", "")
    conclusion_b = facts_b.get("conclusion", "") or paper_b.get("summary", "")
    key_stat_a   = facts_a.get("key_statistic", "Not reported")
    key_stat_b   = facts_b.get("key_statistic", "Not reported")
    study_type_a = facts_a.get("study_type", "Unknown")
    study_type_b = facts_b.get("study_type", "Unknown")

    prompt = f"""You are a senior medical research analyst comparing two research papers.

PAPER A ({paper_a.get('year', '?')} | {paper_a.get('journal', '?')} | Score: {paper_a.get('evidence_score', 0)})
Title     : {paper_a.get('title', '')}
Study type: {study_type_a}
Key stat  : {key_stat_a}
Conclusion: {conclusion_a}
Summary   : {paper_a.get('summary', '')}

PAPER B ({paper_b.get('year', '?')} | {paper_b.get('journal', '?')} | Score: {paper_b.get('evidence_score', 0)})
Title     : {paper_b.get('title', '')}
Study type: {study_type_b}
Key stat  : {key_stat_b}
Conclusion: {conclusion_b}
Summary   : {paper_b.get('summary', '')}

Analyse these two papers carefully. Do their conclusions disagree with each other?

Respond in EXACTLY this format — no extra text:
CONTRADICTION: YES or NO
TYPE: OUTCOME or POPULATION or DOSAGE or METHODOLOGY or NONE
FINDING_A: (one sentence — exact finding of paper A with numbers if available)
FINDING_B: (one sentence — exact finding of paper B with numbers if available)
REASON: (one sentence — the specific reason they contradict)
CLINICAL: (one sentence — what this means for clinical practice)"""

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
    print(f"\nRunning contradiction detection on {len(papers)} papers...")

    if len(papers) < 2:
        print("  Need at least 2 papers.")
        return []

    print("  Computing embeddings...")
    embeddings = get_embeddings(papers)

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
                "paper_a_id"     : paper_a.get("id", ""),
                "paper_b_id"     : paper_b.get("id", ""),
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