from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
import json
import re

client = Groq(api_key=GROQ_API_KEY)


def decompose_query(user_query):
    # Converts plain English medical question into MeSH terms and sub-queries
    # Falls back to raw query if AI response fails to parse

    prompt = f"""You are a medical librarian and PubMed search expert.

Convert this medical question into a precise PubMed search strategy.
Return ONLY a raw JSON object, no markdown, no extra text.

{{
  "condition"   : "main medical condition e.g. Type 2 Diabetes",
  "intervention": "main treatment or drug e.g. Metformin",
  "population"  : "who is studied e.g. elderly adults",
  "mesh_terms"  : ["3-5 precise MeSH terms e.g. Aspirin/therapeutic use"],
  "sub_queries" : ["focused query 1", "focused query 2", "focused query 3"],
  "pubmed_filter": "single best MeSH term as primary filter"
}}

Medical question: "{user_query}"

JSON:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1
        )

        raw   = response.choices[0].message.content.strip()
        raw   = re.sub(r"```json|```", "", raw).strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1

        if start == -1 or end == 0:
            return _fallback(user_query)

        data = json.loads(raw[start:end])

        # Ensure sub_queries is always a list
        if not isinstance(data.get("sub_queries"), list) or not data["sub_queries"]:
            data["sub_queries"] = [user_query]

        print(f"\n[Agent 1 — Query Decomposer]")
        print(f"  Condition   : {data.get('condition', 'Unknown')}")
        print(f"  Intervention: {data.get('intervention', 'Unknown')}")
        print(f"  Population  : {data.get('population', 'General adults')}")
        print(f"  MeSH terms  : {', '.join(data.get('mesh_terms', []))}")
        print(f"  Sub-queries :")
        for i, sq in enumerate(data.get("sub_queries", [])):
            print(f"    {i+1}. {sq}")

        return data

    except Exception as e:
        print(f"  Query decomposer failed: {e} — using fallback")
        return _fallback(user_query)


def _fallback(user_query):
    # Returns minimal structure using raw query if decomposition fails
    return {
        "condition"    : "Unknown",
        "intervention" : "Unknown",
        "population"   : "General adults",
        "mesh_terms"   : [],
        "sub_queries"  : [user_query],
        "pubmed_filter": user_query
    }