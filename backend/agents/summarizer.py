from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
import json
import re

client = Groq(api_key=GROQ_API_KEY)


def _extract_sample_size(value):
    # Converts AI-returned sample size to integer regardless of format
    if value is None:                return 0
    if isinstance(value, int):       return value
    if isinstance(value, float):     return int(value)
    if isinstance(value, str):
        numbers = re.findall(r'\d+', value)
        return int(numbers[0]) if numbers else 0
    return 0


def summarize_abstract(title, abstract):
    # Returns summary, extracted facts, and sample size for one paper
    # Skips AI call if abstract is missing or too short

    if not abstract or abstract == "No abstract":
        return {"summary": "No abstract available.", "facts": {}, "sample_size": 0}

    if len(abstract) < 80:
        return {"summary": "Abstract too short to summarize.", "facts": {}, "sample_size": 0}

    prompt = f"""You are a medical research data extraction assistant.

Read this abstract and return ONLY a raw JSON object — no markdown, no extra text.

{{
  "summary"        : "2-3 sentence plain English summary — what was studied, how, and what was found",
  "sample_size"    : integer number of patients (0 if not mentioned),
  "study_type"     : "exact study design e.g. Randomized Controlled Trial",
  "location"       : "country or region, or Global, or Unknown",
  "key_statistic"  : "single most important number or percentage, or null",
  "drugs_mentioned": "main drugs or treatments, comma separated, or null",
  "conclusion"     : "one plain English sentence — the paper's main conclusion"
}}

Rules:
- sample_size must be a plain integer like 234, never a string
- conclusion must always be filled
- Return raw JSON only

Paper title: {title}
Abstract: {abstract}

JSON:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        # Fix common JSON issues before parsing
        start = raw.find("{")
        end   = raw.rfind("}") + 1

        if start == -1 or end == 0:
            return {"summary": raw[:400], "facts": {}, "sample_size": 0}

        json_str = raw[start:end]
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        data = json.loads(json_str)

        return {
            "summary"    : data.get("summary") or "Summary not available.",
            "facts"      : {
                "study_type"     : data.get("study_type")       or "Unknown",
                "location"       : data.get("location")         or "Unknown",
                "key_statistic"  : data.get("key_statistic")    or "Not reported",
                "drugs_mentioned": data.get("drugs_mentioned")  or "Not mentioned",
                "conclusion"     : data.get("conclusion")       or "Not stated",
            },
            "sample_size": _extract_sample_size(data.get("sample_size", 0))
        }

    except json.JSONDecodeError:
        # Try extracting just the summary if full JSON fails
        match = re.search(r'"summary"\s*:\s*"([^"]+)"', raw)
        summary = match.group(1) if match else "Could not parse summary."
        return {"summary": summary, "facts": {}, "sample_size": 0}

    except Exception as e:
        print(f"    Summarizer error: {e}")
        return {"summary": "Error generating summary.", "facts": {}, "sample_size": 0}


def summarize_all_papers(papers):
    print(f"\nSummarizing {len(papers)} papers...")
    for i, paper in enumerate(papers):
        print(f"  Paper {i+1}/{len(papers)}: {paper['title'][:55]}...")
        result           = summarize_abstract(paper["title"], paper["abstract"])
        paper["summary"]      = result["summary"]
        paper["facts"]        = result["facts"]
        paper["sample_size"]  = result["sample_size"]
    return papers