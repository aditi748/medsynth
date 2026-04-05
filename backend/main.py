from fastapi.responses import StreamingResponse
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from agents.paper_fetcher import fetch_papers
from agents.summarizer import summarize_all_papers
from agents.evidence_scorer import score_papers
from agents.contradiction_detector import detect_contradictions
from agents.report_generator import generate_report

app = FastAPI(
    title="MedSynth API",
    description="Multi-Agent Medical Literature Synthesis System",
    version="1.0.0"
)

# Allows React on port 5173 to talk to this server on port 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    query      : str
    max_results: int = 15


@app.get("/")
def health_check():
    return {
        "status" : "running",
        "message": "MedSynth API is ready",
        "version": "1.0.0"
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query too short.")

    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long.")

    try:
        print(f"\n{'='*60}")
        print(f"Query: {request.query}")
        print(f"{'='*60}")

        papers = fetch_papers(query=request.query, max_results=request.max_results)

        if not papers:
            raise HTTPException(status_code=404, detail="No relevant papers found for this query.")

        papers = summarize_all_papers(papers)
        papers = score_papers(papers)
        contradictions = detect_contradictions(papers)
        report_data = generate_report(
            query=request.query,
            papers=papers,
            contradictions=contradictions
        )

        papers_out = []
        for p in papers:
            papers_out.append({
                "id"             : p.get("id", ""),
                "title"          : p.get("title", ""),
                "year"           : p.get("year", 0),
                "journal"        : p.get("journal", ""),
                "summary"        : p.get("summary", ""),
                "evidence_score" : p.get("evidence_score", 0),
                "relevance_score": p.get("relevance_score", 0),
                "sample_size"    : p.get("sample_size", 0),
                "score_breakdown": p.get("score_breakdown", {}),
                "facts"          : p.get("facts", {}),
                "pub_types"      : p.get("pub_types", []),
            })

        contradictions_out = []
        for c in contradictions:
            contradictions_out.append({
                "type"          : c.get("type", ""),
                "similarity"    : c.get("similarity", 0),
                "paper_a_title" : c.get("paper_a_title", ""),
                "paper_b_title" : c.get("paper_b_title", ""),
                "paper_a_year"  : c.get("paper_a_year", 0),
                "paper_b_year"  : c.get("paper_b_year", 0),
                "paper_a_score" : c.get("paper_a_score", 0),
                "paper_b_score" : c.get("paper_b_score", 0),
                "paper_a_sample": c.get("paper_a_sample", 0),
                "paper_b_sample": c.get("paper_b_sample", 0),
                "finding_a"     : c.get("finding_a", ""),
                "finding_b"     : c.get("finding_b", ""),
                "reason"        : c.get("reason", ""),
                "clinical"      : c.get("clinical", ""),
                "paper_a_id": c.get("paper_a_id", ""),
                "paper_b_id": c.get("paper_b_id", ""),
            })

        return {
            "success"        : True,
            "query"          : request.query,
            "papers"         : papers_out,
            "contradictions" : contradictions_out,
            "report": {
                "verdict"             : report_data.get("verdict", ""),
                "consensus"           : report_data.get("consensus", []),
                "disagreements"       : report_data.get("disagreements", ""),
                "implications"        : report_data.get("implications", ""),
                "confidence"          : report_data.get("confidence", ""),
                "avg_score"           : report_data.get("avg_score", 0),
                "total_patients"      : report_data.get("total_patients", 0),
                "year_range"          : report_data.get("year_range", ""),
                "papers_count"        : report_data.get("papers_count", 0),
                "contradictions_count": report_data.get("contradictions_count", 0),
                "generated_at"        : report_data.get("generated_at", ""),
            },
            "report_text": report_data.get("report_text", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analyze/stream")
async def analyze_stream(query: str, max_results: int = 15):
    """
    Streaming version of /analyze.
    Sends real-time agent status updates as each agent completes.
    Frontend connects using EventSource and listens for events.
    """

    if not query or len(query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query too short.")

    async def event_stream():
        try:
            def send(data: dict):
                # SSE format requires "data: " prefix and double newline
                return f"data: {json.dumps(data)}\n\n"

            # Agent 1 — Query Decomposer
            yield send({"agent": 1, "status": "processing", "message": "Decomposing query into MeSH terms..."})
            await asyncio.sleep(0)  # lets FastAPI flush the response

            papers_list = []
            decomposed  = None

            from agents.query_decomposer import decompose_query
            decomposed = decompose_query(query)
            yield send({"agent": 1, "status": "done", "message": "Query decomposed into MeSH terms and sub-queries"})

            # Agent 2 — Paper Fetcher
            yield send({"agent": 2, "status": "processing", "message": "Searching PubMed and ArXiv..."})
            await asyncio.sleep(0)

            from agents.paper_fetcher import fetch_papers
            papers_list = fetch_papers(query=query, max_results=max_results)

            if not papers_list:
                yield send({"type": "error", "message": "No relevant papers found for this query."})
                return

            yield send({"agent": 2, "status": "done", "message": f"{len(papers_list)} relevant papers fetched"})

            # Agent 3 — Summarizer
            yield send({"agent": 3, "status": "processing", "message": f"Summarising {len(papers_list)} papers..."})
            await asyncio.sleep(0)

            from agents.summarizer import summarize_all_papers
            papers_list = summarize_all_papers(papers_list)
            yield send({"agent": 3, "status": "done", "message": "All abstracts summarised with facts extracted"})

            # Agent 5 — Evidence Scorer
            yield send({"agent": 5, "status": "processing", "message": "Scoring papers by evidence quality..."})
            await asyncio.sleep(0)

            from agents.evidence_scorer import score_papers
            papers_list = score_papers(papers_list)
            yield send({"agent": 5, "status": "done", "message": "Evidence scores computed"})

            # Agent 4 — Contradiction Detector
            yield send({"agent": 4, "status": "processing", "message": "Detecting contradictions between papers..."})
            await asyncio.sleep(0)

            from agents.contradiction_detector import detect_contradictions
            contradictions = detect_contradictions(papers_list)
            yield send({"agent": 4, "status": "done", "message": f"{len(contradictions)} contradiction(s) detected"})

            # Agent 6 — Report Generator
            yield send({"agent": 6, "status": "processing", "message": "Generating structured report..."})
            await asyncio.sleep(0)

            from agents.report_generator import generate_report
            report_data = generate_report(
                query=query,
                papers=papers_list,
                contradictions=contradictions
            )
            yield send({"agent": 6, "status": "done", "message": "Report generated"})

            # Build final result — same structure as /analyze
            papers_out = []
            for p in papers_list:
                papers_out.append({
                    "id"             : p.get("id", ""),
                    "title"          : p.get("title", ""),
                    "year"           : p.get("year", 0),
                    "journal"        : p.get("journal", ""),
                    "summary"        : p.get("summary", ""),
                    "evidence_score" : p.get("evidence_score", 0),
                    "relevance_score": p.get("relevance_score", 0),
                    "sample_size"    : p.get("sample_size", 0),
                    "score_breakdown": p.get("score_breakdown", {}),
                    "facts"          : p.get("facts", {}),
                    "pub_types"      : p.get("pub_types", []),
                })

            contradictions_out = []
            for c in contradictions:
                contradictions_out.append({
                    "type"          : c.get("type", ""),
                    "similarity"    : c.get("similarity", 0),
                    "paper_a_title" : c.get("paper_a_title", ""),
                    "paper_b_title" : c.get("paper_b_title", ""),
                    "paper_a_year"  : c.get("paper_a_year", 0),
                    "paper_b_year"  : c.get("paper_b_year", 0),
                    "paper_a_score" : c.get("paper_a_score", 0),
                    "paper_b_score" : c.get("paper_b_score", 0),
                    "paper_a_sample": c.get("paper_a_sample", 0),
                    "paper_b_sample": c.get("paper_b_sample", 0),
                    "finding_a"     : c.get("finding_a", ""),
                    "finding_b"     : c.get("finding_b", ""),
                    "reason"        : c.get("reason", ""),
                    "clinical"      : c.get("clinical", ""),
                    "paper_a_id": c.get("paper_a_id", ""),
                    "paper_b_id": c.get("paper_b_id", ""),
                })

            # Send complete result as final event
            yield send({
                "type"          : "complete",
                "success"       : True,
                "query"         : query,
                "papers"        : papers_out,
                "contradictions": contradictions_out,
                "report": {
                    "verdict"             : report_data.get("verdict", ""),
                    "consensus"           : report_data.get("consensus", []),
                    "disagreements"       : report_data.get("disagreements", ""),
                    "implications"        : report_data.get("implications", ""),
                    "confidence"          : report_data.get("confidence", ""),
                    "avg_score"           : report_data.get("avg_score", 0),
                    "total_patients"      : report_data.get("total_patients", 0),
                    "year_range"          : report_data.get("year_range", ""),
                    "papers_count"        : report_data.get("papers_count", 0),
                    "contradictions_count": report_data.get("contradictions_count", 0),
                    "generated_at"        : report_data.get("generated_at", ""),
                },
                "report_text": report_data.get("report_text", ""),
            })

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",
            "Access-Control-Allow-Origin": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)