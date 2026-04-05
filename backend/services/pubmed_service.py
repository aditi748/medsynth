import requests
import xml.etree.ElementTree as ET
import re

MIN_YEAR            = 2000  # widened from 2010
MAX_YEAR            = 2024
MIN_ABSTRACT_LENGTH = 80    # lowered from 150

def build_search_query(query, mesh_terms=None, include_pub_filter=False):
    # Builds PubMed query
    # Publication type filter is now OPTIONAL — only used for strict clinical queries
    # Removing it by default multiplies available papers by 10x+

    if mesh_terms and len(mesh_terms) > 0:
        mesh_parts = [f'"{term}"[MeSH Terms]' for term in mesh_terms[:3]]
        topic      = "(" + " AND ".join(mesh_parts) + ")"
        print(f"  Using MeSH terms: {', '.join(mesh_terms[:3])}")
    else:
        topic = f"({query})"
        print(f"  Using keyword search: {query}")

    query_parts = [
        topic,
        f"{MIN_YEAR}:{MAX_YEAR}[dp]",
        "english[lang]",
        "hasabstract",
    ]

    if include_pub_filter:
        # Only used for strict clinical queries when explicitly requested
        CLINICAL_TYPES = [
            "Randomized Controlled Trial",
            "Meta-Analysis",
            "Systematic Review",
            "Clinical Trial",
            "Multicenter Study",
        ]
        pub_filter = " OR ".join([f'"{pt}"[pt]' for pt in CLINICAL_TYPES])
        query_parts.append(f"({pub_filter})")

    return " AND ".join(query_parts)


def search_pubmed(query, max_results=20, mesh_terms=None, strict=False):
    # Searches PubMed — now fetches up to 60 candidates before filtering
    # strict=True adds publication type filter for clinical drug queries
    print(f"\n  Searching PubMed for: '{query}'")

    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db"        : "pubmed",
            "term"      : build_search_query(query, mesh_terms, include_pub_filter=strict),
            "retmax"    : max_results * 3,  # fetch 3x, filter down
            "retmode"   : "json",
            "sort"      : "relevance",
            "usehistory": "n",
        },
        timeout=15
    )

    if response.status_code != 200:
        print(f"  PubMed search failed: {response.status_code}")
        return []

    paper_ids = response.json()["esearchresult"]["idlist"]
    print(f"  PubMed returned {len(paper_ids)} IDs")

    if not paper_ids:
        return []

    papers = fetch_paper_details(paper_ids)
    papers = filter_papers(papers, max_results)
    print(f"  {len(papers)} papers passed quality filter.")
    return papers


def fetch_paper_details(paper_ids):
    response = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={
            "db"     : "pubmed",
            "id"     : ",".join(paper_ids),
            "retmode": "xml",
        },
        timeout=20
    )
    if response.status_code != 200:
        return []
    return parse_pubmed_xml(response.text)


def parse_pubmed_xml(xml_text):
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            title_el    = article.find(".//ArticleTitle")
            abstract_el = article.find(".//Abstract")
            year_el     = article.find(".//PubDate/Year")
            journal_el  = article.find(".//Journal/Title")
            pmid_el     = article.find(".//PMID")

            title    = "".join(title_el.itertext()).strip()     if title_el    is not None else ""
            abstract = " ".join(abstract_el.itertext()).strip() if abstract_el is not None else ""
            journal  = journal_el.text.strip()                  if journal_el  is not None else "Unknown"
            pmid     = pmid_el.text.strip()                     if pmid_el     is not None else "0"

            if year_el is not None:
                year = int(year_el.text)
            else:
                medline_el = article.find(".//MedlineDate")
                year = int(medline_el.text[:4]) if medline_el is not None else 2000

            pub_types = [
                pt.text.strip()
                for pt in article.findall(".//PublicationType")
                if pt.text
            ]

            papers.append({
                "id"         : pmid,
                "title"      : title,
                "abstract"   : abstract,
                "year"       : year,
                "journal"    : journal,
                "pub_types"  : pub_types,
                "citations"  : 0,
                "sample_size": 0,
                "source"     : "pubmed",
            })
        except Exception:
            continue

    return papers


def filter_papers(papers, max_results):
    # Minimal quality filter — only removes genuinely unusable papers
    quality = []
    for paper in papers:
        if not paper["title"] or len(paper["title"]) < 10:
            continue
        if not paper["abstract"] or len(paper["abstract"]) < MIN_ABSTRACT_LENGTH:
            continue
        if not (MIN_YEAR <= paper["year"] <= MAX_YEAR):
            continue
        quality.append(paper)
        if len(quality) >= max_results:
            break
    return quality


def extract_sample_size(abstract):
    # Pattern matching for common sample size expressions
    if not abstract:
        return 0

    patterns = [
        r'n\s*=\s*([\d,]+)',
        r'n\s*=\s*([\d,]+)\s*(?:patients|participants|subjects|adults|women|men|children)',
        r'([\d,]+)\s+patients\s+were\s+(?:randomized|enrolled|included|recruited|assigned)',
        r'([\d,]+)\s+participants\s+were\s+(?:randomized|enrolled|included|recruited)',
        r'enrolled\s+([\d,]+)\s+(?:patients|participants|subjects|adults)',
        r'recruited\s+([\d,]+)\s+(?:patients|participants|subjects|adults)',
        r'total\s+of\s+([\d,]+)\s+(?:patients|participants|subjects)',
        r'included\s+([\d,]+)\s+(?:patients|participants|subjects)',
        r'comprising\s+([\d,]+)\s+(?:patients|participants|subjects)',
        r'among\s+([\d,]+)\s+(?:patients|participants|subjects)',
        r'([\d,]+)\s+(?:patients|participants|subjects)\s+(?:were|with|who)',
        r'cohort\s+of\s+([\d,]+)',
        r'sample\s+of\s+([\d,]+)',
        r'([\d,]+)\s+(?:adults|women|men|children|elderly|individuals)\s+(?:were|with|who)',
    ]

    for pattern in patterns:
        match = re.search(pattern, abstract, re.IGNORECASE)
        if match:
            try:
                size = int(match.group(1).replace(",", ""))
                if 10 <= size <= 1_000_000:
                    return size
            except ValueError:
                continue
    return 0