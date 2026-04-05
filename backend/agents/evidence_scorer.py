from datetime import datetime
import math

CURRENT_YEAR     = datetime.now().year
OLDEST_PAPER_AGE = 20

# Journal prestige scores
# Floor is 0.45 — any indexed peer-reviewed journal has already been vetted
# Unknown journals get 0.45, not 0.30
JOURNAL_PRESTIGE = {
    "the new england journal of medicine"            : 1.0,
    "the lancet"                                     : 1.0,
    "jama"                                           : 1.0,
    "the journal of the american medical association": 1.0,
    "nature medicine"                                : 1.0,
    "bmj"                                            : 1.0,
    "british medical journal"                        : 1.0,
    "cochrane database of systematic reviews"        : 0.95,
    "annals of internal medicine"                    : 0.88,
    "circulation"                                    : 0.88,
    "journal of the american college of cardiology"  : 0.88,
    "european heart journal"                         : 0.88,
    "diabetes care"                                  : 0.88,
    "chest"                                          : 0.85,
    "gut"                                            : 0.85,
    "journal of clinical oncology"                   : 0.85,
    "annals of oncology"                             : 0.85,
    "american journal of respiratory and critical care medicine": 0.85,
    "plos medicine"                                  : 0.75,
    "plos one"                                       : 0.72,
    "bmc medicine"                                   : 0.72,
    "scientific reports"                             : 0.70,
    "american journal of epidemiology"               : 0.70,
    "journal of internal medicine"                   : 0.70,
    "international journal of epidemiology"          : 0.70,
    "frontiers in medicine"                          : 0.60,
    "frontiers in cardiovascular medicine"           : 0.58,
    "frontiers in endocrinology"                     : 0.58,
    "frontiers in nutrition"                         : 0.58,
    "bmc cardiovascular disorders"                   : 0.58,
    "bmc public health"                              : 0.60,
    "nutrients"                                      : 0.62,
    "nutrition journal"                              : 0.60,
    "public health nutrition"                        : 0.65,
}
DEFAULT_PRESTIGE = 0.45 

# Study type scores — given highest weight because study design
# is the most important clinical indicator of reliability
STUDY_TYPE_SCORES = {
    "meta-analysis"                    : 1.0,
    "systematic review"                : 0.92,
    "randomized controlled trial"      : 0.88,
    "controlled clinical trial"        : 0.78,
    "multicenter study"                : 0.75,
    "clinical trial"                   : 0.70,
    "cohort study"                     : 0.62,
    "prospective study"                : 0.60,
    "longitudinal study"               : 0.58,
    "observational study"              : 0.50,
    "cross-sectional study"            : 0.48,
    "case-control study"               : 0.45,
    "review"                           : 0.42,
    "narrative review"                 : 0.38,
    "preprint"                         : 0.22,
}
DEFAULT_STUDY_TYPE_SCORE = 0.40  # raised from 0.30


def get_journal_prestige(journal_name):
    # Returns prestige score — partial matching handles variations
    if not journal_name:
        return DEFAULT_PRESTIGE
    journal_lower = journal_name.lower().strip()
    for known_journal, score in JOURNAL_PRESTIGE.items():
        if known_journal in journal_lower or journal_lower in known_journal:
            return score
    return DEFAULT_PRESTIGE


def get_study_type_score(pub_types):
    # Returns highest trust score found in the paper's study type list
    if not pub_types:
        return DEFAULT_STUDY_TYPE_SCORE
    best_score = DEFAULT_STUDY_TYPE_SCORE
    for pt in pub_types:
        pt_lower = pt.lower().strip()
        for known_type, score in STUDY_TYPE_SCORES.items():
            if known_type in pt_lower:
                best_score = max(best_score, score)
    return best_score


def normalize_sample_size(sample_size, max_sample):
    # Logarithmic scale — the jump from 50 to 500 patients matters far more
    # than the jump from 50,000 to 100,000. Linear scale was unfairly
    # crushing smaller studies when one outlier had a huge sample.
    if sample_size <= 0:
        return 0.0
    if max_sample <= 0:
        return 0.0
    # log scale: score = log(sample+1) / log(max_sample+1)
    score = math.log(sample_size + 1) / math.log(max_sample + 1)
    return min(1.0, score)


def compute_recency_score(year):
    # Last 2 years = 1.0, older than 20 years = 0.0, linear in between
    age = CURRENT_YEAR - year
    if age <= 2:
        return 1.0
    if age >= OLDEST_PAPER_AGE:
        return 0.0
    return 1.0 - ((age - 2) / (OLDEST_PAPER_AGE - 2))


def score_papers(papers):
    # Revised formula — study type carries most weight as per evidence-based medicine
    # Score = (StudyType×0.30) + (Recency×0.25) + (Prestige×0.25) + (Sample×0.20)
    
    # Find max sample size for logarithmic normalisation
    sample_values = [p.get("sample_size", 0) for p in papers]
    max_sample    = max(sample_values) if sample_values else 1

    for paper in papers:
        study_type_score = get_study_type_score(paper.get("pub_types", []))
        recency_score    = compute_recency_score(paper.get("year", 2000))
        prestige_score   = get_journal_prestige(paper.get("journal", ""))
        sample_score     = normalize_sample_size(
            paper.get("sample_size", 0), max_sample
        )

        evidence_score = (
            (study_type_score * 0.30) +
            (recency_score    * 0.25) +
            (prestige_score   * 0.25) +
            (sample_score     * 0.20)
        )

        paper["evidence_score"]  = round(evidence_score, 2)
        paper["score_breakdown"] = {
            "study_type"      : round(study_type_score, 2),
            "recency"         : round(recency_score, 2),
            "journal_prestige": round(prestige_score, 2),
            "sample_size"     : round(sample_score, 2),
        }

    papers.sort(key=lambda p: p["evidence_score"], reverse=True)
    return papers