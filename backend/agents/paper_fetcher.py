from services.pubmed_service import search_pubmed, extract_sample_size
from services.arxiv_service import search_arxiv
from agents.query_decomposer import decompose_query
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
import time
import re

client = Groq(api_key=GROQ_API_KEY)

MIN_PAPERS          = 15   # always try to return at least this many
HARD_DISCARD_SCORE  = 2    # only discard papers scoring 0-2 (clearly off-topic)


def is_health_related(query):
    # Single upfront check — is this query related to health, medicine,
    # biology, nutrition, or science at all?
    # Only returns False for clearly unrelated queries like restaurants or sports scores.
    prompt = f"""Is this query related to health, medicine, biology, nutrition, 
pharmacology, disease, treatment, or any life science topic?

Query: "{query}"

Answer with only YES or NO."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception:
        return True  # if check fails, proceed anyway


def score_relevance(query, title, abstract):
    # Two-stage relevance check
    # Stage 1: fast keyword check
    query_words  = [w.lower() for w in query.split() if len(w) > 3]
    combined     = (title + " " + (abstract or "")).lower()
    matches      = sum(1 for w in query_words if w in combined)

    # If zero query words appear anywhere, almost certainly irrelevant
    if matches == 0:
        return 0

    # Stage 2: AI scores relevance
    text = abstract[:500] if abstract and len(abstract) > 100 else title

    prompt = f"""Rate how relevant this paper is to the research query.

Query: "{query}"
Title: {title}
Text : {text}

Score 0-10:
0-2  = completely unrelated topic
3-5  = loosely related but not about this query
6-7  = relevant, discusses related topic
8-10 = directly answers the query

Return ONLY a single integer."""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0
        )
        raw     = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', raw)
        return max(0, min(10, int(numbers[0]))) if numbers else 5
    except Exception:
        return 5


def fetch_papers(query, max_results=15):
    # Agent 2 — fetches papers from all sources and filters by relevance
    # Always tries to return at least MIN_PAPERS papers

    # Step 1: Check if query is health-related at all
    print(f"\n[Agent 2 — Paper Fetcher]")
    print(f"  Checking topic relevance...")

    if not is_health_related(query):
        print("  Query is not health/medicine related — stopping.")
        return []

    # Step 2: Decompose query into MeSH terms and sub-queries
    decomposed  = decompose_query(query)
    mesh_terms  = decomposed.get("mesh_terms", [])
    sub_queries = decomposed.get("sub_queries", [query])

    all_papers = []
    seen_ids   = set()

    def add_papers(new_papers):
        for p in new_papers:
            if p["id"] not in seen_ids:
                all_papers.append(p)
                seen_ids.add(p["id"])

    # Step 3: MeSH search (most precise)
    if mesh_terms:
        print(f"\n  MeSH search...")
        add_papers(search_pubmed(query, max_results=20, mesh_terms=mesh_terms))

    # Step 4: Sub-query searches
    print(f"\n  Sub-query searches...")
    for i, sq in enumerate(sub_queries[:3]):
        print(f"  Sub-query {i+1}: '{sq}'")
        add_papers(search_pubmed(sq, max_results=15))

    # Step 5: If still not enough papers, try broader keyword search
    if len(all_papers) < MIN_PAPERS:
        print(f"\n  Only {len(all_papers)} papers found — trying broader search...")
        # Extract core keywords from query (2-3 most important words)
        words        = [w for w in query.lower().split() if len(w) > 4]
        broad_query  = " ".join(words[:3])
        add_papers(search_pubmed(broad_query, max_results=20))

    # Step 6: ArXiv for recent preprints
    print(f"\n  ArXiv search...")
    add_papers(search_arxiv(query, max_results=5))

    print(f"\n  Total unique papers before filtering: {len(all_papers)}")

    if not all_papers:
        return []

    # Step 7: Extract sample sizes
    print(f"\n  Extracting sample sizes...")
    for paper in all_papers:
        size = extract_sample_size(paper["abstract"])
        paper["sample_size"] = size
        if size > 0:
            print(f"    n={size}: {paper['title'][:50]}...")

    # Step 8: Score relevance — ranker not gatekeeper
    # Only hard-discard papers scoring 0-2 (clearly off-topic)
    # Everything else stays and gets ranked
    print(f"\n  Scoring relevance of {len(all_papers)} papers...")

    scored_papers = []
    for paper in all_papers:
        score              = score_relevance(query, paper["title"], paper["abstract"])
        paper["relevance_score"] = score

        if score <= HARD_DISCARD_SCORE:
            print(f"    [{score}/10] REMOVED — {paper['title'][:55]}...")
        else:
            scored_papers.append(paper)
            print(f"    [{score}/10] KEPT    — {paper['title'][:55]}...")

        time.sleep(0.2)

    # Step 9: Sort by relevance score, return top max_results
    scored_papers.sort(key=lambda p: p["relevance_score"], reverse=True)
    final_papers = scored_papers[:max_results]

    print(f"\n  Returning {len(final_papers)} papers.")
    return final_papers