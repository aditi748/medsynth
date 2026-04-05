import requests
import xml.etree.ElementTree as ET
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def search_arxiv(query, max_results=3):
    # ArXiv terms of service require a delay between requests
    # For a demo, a small sleep ensures we don't get 429 errors
    time.sleep(2) 
    
    # 1. Setup a Retry Strategy (Professional approach for CSE projects)
    session = requests.Session()
    retry_strategy = Retry(
        total=3, # Retry 3 times
        backoff_factor=2, # Wait 2s, 4s, 8s between retries
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

    try:
        # 2. Increase Timeout: (Connect timeout, Read timeout)
        # Using 30s for read to handle slow scientific database responses
        response = session.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            timeout=(5, 30) 
        )

        if response.status_code != 200:
            print(f"    ArXiv request failed: {response.status_code}")
            return []

        return parse_arxiv_xml(response.text)

    except requests.exceptions.Timeout:
        print("    ArXiv error: Connection timed out. ArXiv is likely under heavy load.")
        return []
    except Exception as e:
        print(f"    ArXiv error: {e}")
        return []

def parse_arxiv_xml(xml_text):
    # Parses ArXiv XML — uses atom namespace unlike PubMed
    papers = []
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"    ArXiv XML parse error: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", ns):
        try:
            title_el     = entry.find("atom:title", ns)
            abstract_el  = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el        = entry.find("atom:id", ns)

            # Clean up whitespace and newlines often found in ArXiv abstracts
            title    = " ".join(title_el.text.strip().split())     if title_el     is not None else "No title"
            abstract = " ".join(abstract_el.text.strip().split()) if abstract_el  is not None else "No abstract"
            year     = int(published_el.text[:4])                  if published_el is not None else 2026
            arxiv_id = id_el.text.strip()                          if id_el        is not None else "0"

            papers.append({
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": "ArXiv (preprint)",
                "pub_types": ["Preprint"],
                "citations": 0,
                "sample_size": 0,
                "source": "arxiv",
            })

        except Exception as e:
            print(f"    Skipped ArXiv paper: {e}")
            continue

    return papers