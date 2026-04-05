from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from datetime import datetime

client = Groq(api_key=GROQ_API_KEY)


def generate_report(query, papers, contradictions, decomposed_query=None):
    # Agent 6 — reads all agent outputs and produces the final structured report
    print("\n[Agent 6 — Report Generator]")
    print("  Generating report...")

    # Build context from top 5 papers for the AI
    papers_context = ""
    for i, p in enumerate(papers[:5]):
        facts = p.get("facts", {})
        papers_context += f"""
Paper {i+1} (Score: {p['evidence_score']} | Year: {p['year']})
Title     : {p['title']}
Study type: {facts.get('study_type', 'Unknown')}
Sample    : {p.get('sample_size', 0)} participants
Key stat  : {facts.get('key_statistic', 'Not reported')}
Conclusion: {facts.get('conclusion', 'Not stated')}
Summary   : {p['summary']}
"""

    # Build contradiction context
    contradictions_context = ""
    if contradictions:
        for i, c in enumerate(contradictions):
            contradictions_context += f"""
Contradiction {i+1} ({c['type']}):
Paper A: {c['paper_a_title']} ({c['paper_a_year']}) — {c.get('finding_a', 'N/A')}
Paper B: {c['paper_b_title']} ({c['paper_b_year']}) — {c.get('finding_b', 'N/A')}
Reason : {c.get('reason', 'N/A')}
"""
    else:
        contradictions_context = "No contradictions detected."

    # Calculate summary metrics
    total_patients = sum(p.get('sample_size', 0) for p in papers)
    years          = [p['year'] for p in papers]
    year_range     = f"{min(years)}–{max(years)}"
    # Calculate based on top 5 papers and proportion of high trust papers
    top_5        = papers[:5]
    avg_top5     = round(sum(p['evidence_score'] for p in top_5) / len(top_5), 2)
    high_trust   = [p for p in papers if p['evidence_score'] >= 0.70]
    total        = len(papers)
    high_ratio   = len(high_trust) / total  # proportion of high quality papers

    avg_score    = round(sum(p['evidence_score'] for p in papers) / total, 2)

    if avg_top5 >= 0.75 and len(high_trust) >= 3:
        confidence        = "HIGH"
        confidence_reason = f"{len(high_trust)} high-quality studies analysed with strong evidence"
    elif avg_top5 >= 0.60 or (len(high_trust) >= 2 and high_ratio >= 0.2):
        confidence        = "MODERATE"
        confidence_reason = "Mix of high and moderate quality evidence available"
    else:
        confidence        = "LOW"
        confidence_reason = "Limited high-quality evidence — mostly reviews and older studies"

    # Ask AI to write verdict, consensus, disagreements, implications
    prompt = f"""You are a senior medical researcher writing a literature review.

Research question: "{query}"

Papers analysed  : {len(papers)}
Date range       : {year_range}
Total patients   : {total_patients:,}
Avg trust score  : {avg_score}/1.0
Confidence level : {confidence}

Top papers:
{papers_context}

Contradictions:
{contradictions_context}

Write exactly these four sections:

VERDICT: (2-3 sentences directly answering the research question. Be specific. Use numbers.)

CONSENSUS: (3 key findings most papers agree on. Each as one sentence starting with -)

DISAGREEMENTS: (2-3 sentences on where papers contradict and why it matters. If none, say evidence is largely consistent.)

IMPLICATIONS: (2 sentences on what this means for clinical practice — who should or should not receive this treatment.)

Rules:
- Be specific and evidence-based
- Use numbers from papers where available
- Write for a medical professional audience"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.1
        )
        ai_content = response.choices[0].message.content.strip()
        sections   = _parse_sections(ai_content)

    except Exception as e:
        print(f"  Report generation error: {e}")
        sections = {
            "verdict"      : "Could not generate verdict.",
            "consensus"    : [],
            "disagreements": "Could not analyse disagreements.",
            "implications" : "Consult a medical professional."
        }

    report_text = _build_report(
        query, papers, contradictions, sections,
        confidence, confidence_reason,
        avg_score, total_patients, year_range
    )

    return {
        "report_text"         : report_text,
        "verdict"             : sections.get("verdict", ""),
        "consensus"           : sections.get("consensus", []),
        "disagreements"       : sections.get("disagreements", ""),
        "implications"        : sections.get("implications", ""),
        "confidence"          : confidence,
        "avg_score"           : avg_score,
        "total_patients"      : total_patients,
        "year_range"          : year_range,
        "papers_count"        : len(papers),
        "contradictions_count": len(contradictions),
        "generated_at"        : datetime.now().strftime("%Y-%m-%d %H:%M")
    }


def _parse_sections(ai_text):
    # Parses AI response into verdict, consensus, disagreements, implications
    sections = {"verdict": "", "consensus": [], "disagreements": "", "implications": ""}
    ai_text  = ai_text.replace("**", "").replace("##", "").strip()
    lines    = [l.strip() for l in ai_text.split("\n") if l.strip()]

    current        = None
    consensus_lines = []

    for line in lines:
        upper = line.upper()

        if "VERDICT" in upper and len(line) < 60:
            current = "verdict"
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            if content:
                sections["verdict"] = content
            continue
        elif "CONSENSUS" in upper and len(line) < 60:
            current = "consensus"
            continue
        elif "DISAGREEMENT" in upper and len(line) < 60:
            current = "disagreements"
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            if content:
                sections["disagreements"] = content
            continue
        elif "IMPLICATION" in upper and len(line) < 60:
            current = "implications"
            content = line.split(":", 1)[1].strip() if ":" in line else ""
            if content:
                sections["implications"] = content
            continue

        if current == "verdict":
            sections["verdict"] += (" " + line) if sections["verdict"] else line
        elif current == "consensus":
            clean = line.lstrip("-•123456789. ").strip()
            if clean and len(clean) > 10:
                consensus_lines.append(clean)
        elif current == "disagreements":
            sections["disagreements"] += (" " + line) if sections["disagreements"] else line
        elif current == "implications":
            sections["implications"] += (" " + line) if sections["implications"] else line

    sections["consensus"] = consensus_lines[:5]
    return sections


def _build_report(query, papers, contradictions, sections,
                   confidence, confidence_reason,
                   avg_score, total_patients, year_range):
    # Assembles the final formatted report string

    div  = "=" * 70
    thin = "-" * 70

    badge = {
        "HIGH"    : "🟢 HIGH CONFIDENCE",
        "MODERATE": "🟡 MODERATE CONFIDENCE",
        "LOW"     : "🔴 LOW CONFIDENCE"
    }.get(confidence, "⚪ UNKNOWN")

    report = f"""
{div}
MEDSYNTH LITERATURE REVIEW REPORT
{div}
Generated : {datetime.now().strftime("%Y-%m-%d %H:%M")}
Query     : {query}
{div}

SEARCH SUMMARY
{thin}
Papers analysed : {len(papers)}
Date range      : {year_range}
Total patients  : {total_patients:,}
Avg trust score : {avg_score}/1.0
Contradictions  : {len(contradictions)} detected
Confidence      : {badge}
Reason          : {confidence_reason}

{div}
SECTION 1 — OVERALL VERDICT
{div}
{sections.get('verdict', 'Could not generate verdict.')}

{div}
SECTION 2 — KEY CONSENSUS FINDINGS
{div}
What most papers agree on:
"""

    consensus = sections.get("consensus", [])
    if consensus:
        for point in consensus:
            report += f"  • {point}\n"
    else:
        report += "  No clear consensus identified.\n"

    report += f"""
{div}
SECTION 3 — AREAS OF DISAGREEMENT
{div}
{sections.get('disagreements', 'No significant disagreements detected.')}
"""

    if contradictions:
        report += "\nDetailed contradictions:\n"
        for i, c in enumerate(contradictions):
            report += f"""
  Contradiction #{i+1} [{c['type']}]
  Paper A ({c['paper_a_year']}): {c['paper_a_title'][:65]}
           Finding: {c.get('finding_a', 'N/A')}
  Paper B ({c['paper_b_year']}): {c['paper_b_title'][:65]}
           Finding: {c.get('finding_b', 'N/A')}
  Reason : {c.get('reason', 'N/A')}
  Clinical significance: {c.get('clinical', 'N/A')}
"""

    report += f"""
{div}
SECTION 4 — CLINICAL IMPLICATIONS
{div}
{sections.get('implications', 'Consult a medical professional for clinical decisions.')}

{div}
SECTION 5 — LIMITATIONS AND DISCLAIMER
{div}
  • Based on {len(papers)} papers — not a comprehensive systematic review
  • Papers span {year_range} — very recent evidence may be missing
  • Evidence scores based on journal prestige, recency, and sample size
  • For RESEARCH ASSISTANCE ONLY — not a substitute for clinical judgement
  • Always consult a qualified healthcare professional

{div}
TOP PAPERS BY EVIDENCE SCORE
{div}"""

    for i, paper in enumerate(papers[:5]):
        score = paper['evidence_score']
        bar   = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        report += f"""
{i+1}. [{bar}] {score}
   {paper['title']}
   {paper['journal']} ({paper['year']}) | Sample: {paper.get('sample_size', 0):,}
   {paper['summary'][:200]}...
"""

    report += f"\n{div}\n"
    report += "END OF REPORT — MedSynth v1.0\n"
    report += f"{div}\n"

    return report