
import re
import json
from collections import Counter
from urllib.parse import urlparse


# ---------------------------------------------------------
# 1. SOURCE REFERENCE DATA
# These are heuristic reference scores, not verified ratings.
# ---------------------------------------------------------

SOURCE_CREDIBILITY = {
    "reuters.com": 98,
    "apnews.com": 98,
    "bbc.com": 95,
    "bbc.co.uk": 95,
    "npr.org": 94,
    "pbs.org": 94,
    "snopes.com": 93,
    "politifact.com": 93,
    "factcheck.org": 93,
    "nytimes.com": 92,
    "washingtonpost.com": 92,
    "abcnews.go.com": 91,
    "nbcnews.com": 91,
    "cbsnews.com": 91,
    "theguardian.com": 90,
    "usatoday.com": 88,
    "firstpost.com": 80,
    "businessinsider.com": 82,
    "wikipedia.org": 55,
    "youtube.com": 55,
    "facebook.com": 45,
    "x.com": 35,
}

SOURCE_CATEGORIES = {
    "News Agency": [
        "reuters.com",
        "apnews.com",
    ],
    "News Publisher": [
        "bbc.com",
        "bbc.co.uk",
        "npr.org",
        "pbs.org",
        "nytimes.com",
        "washingtonpost.com",
        "abcnews.go.com",
        "nbcnews.com",
        "cbsnews.com",
        "theguardian.com",
        "usatoday.com",
        "firstpost.com",
        "businessinsider.com",
    ],
    "Fact Checking Organization": [
        "snopes.com",
        "politifact.com",
        "factcheck.org",
    ],
    "Reference Platform": [
        "wikipedia.org",
    ],
    "Social Media": [
        "youtube.com",
        "facebook.com",
        "x.com",
    ],
}

# Factor weights
FACTOR_WEIGHTS = {
    "publisher_reputation": 35,
    "source_transparency": 20,
    "evidence_quality": 25,
    "publication_transparency": 10,
    "source_independence": 10,
}


# ---------------------------------------------------------
# 2. DOMAIN UTILITIES
# ---------------------------------------------------------

def get_domain(url):
    """Extract a normalized domain from a URL."""

    if not url:
        return "unknown"

    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        domain = parsed.netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain or "unknown"

    except Exception:
        return "unknown"


def get_root_publisher(domain):
    """Match subdomains with their known publisher domain."""

    for known_domain in SOURCE_CREDIBILITY:
        if domain == known_domain or domain.endswith("." + known_domain):
            return known_domain

    for category_domains in SOURCE_CATEGORIES.values():
        for known_domain in category_domains:
            if domain == known_domain or domain.endswith("." + known_domain):
                return known_domain

    return domain


def get_credibility(domain):
    """Return a reference score, or None if not assessed."""

    publisher = get_root_publisher(domain)
    return SOURCE_CREDIBILITY.get(publisher)


def get_source_category(domain):
    """Classify a publisher using its domain."""

    publisher = get_root_publisher(domain)

    for category, domains in SOURCE_CATEGORIES.items():
        if publisher in domains:
            return category

    return "Other"


# ---------------------------------------------------------
# 3. FACTOR ASSESSMENT
# ---------------------------------------------------------

def assess_source_transparency(result):
    """
    Assess visible source transparency indicators.
    This does not verify ownership or editorial policies.
    """

    content = str(result.get("content", "") or "").lower()
    title = str(result.get("title", "") or "").lower()

    score = 50
    indicators = []

    author_patterns = [
        r"\bby [a-z][a-z .'-]{2,40}",
        r"\breported by\b",
        r"\bwritten by\b",
        r"\bauthor:",
    ]

    if any(re.search(pattern, content + " " + title)
           for pattern in author_patterns):
        score += 25
        indicators.append("Possible author attribution found")

    if result.get("published_date"):
        score += 25
        indicators.append("Publication date available")

    return min(score, 100), indicators


def assess_evidence_quality(result):
    """
    Estimate evidence indicators from the search result text.
    This does not independently validate the claims or citations.
    """

    content = str(result.get("content", "") or "")
    lower_content = content.lower()

    score = 40
    indicators = []

    # Attribution and reporting indicators
    attribution_terms = [
        "according to",
        "said",
        "reported",
        "statement",
        "officials",
        "researchers",
        "study",
        "data",
    ]

    attribution_count = sum(
        1 for term in attribution_terms if term in lower_content
    )

    if attribution_count >= 3:
        score += 25
        indicators.append("Multiple attribution/reporting indicators")
    elif attribution_count >= 1:
        score += 15
        indicators.append("Attribution/reporting indicator found")

    # Citation or reference indicators
    if re.search(r"https?://|doi\.org|references|source:", lower_content):
        score += 20
        indicators.append("Possible citation or reference found")

    # Content availability
    if len(content.strip()) >= 300:
        score += 15
        indicators.append("Substantial search-result content available")
    elif len(content.strip()) >= 100:
        score += 8
        indicators.append("Some search-result content available")

    return min(score, 100), indicators


def assess_publication_transparency(result):
    """Assess availability of author and publication-date metadata."""

    score = 0
    indicators = []

    content = str(result.get("content", "") or "")
    title = str(result.get("title", "") or "")

    if result.get("published_date"):
        score += 50
        indicators.append("Publication date available")

    author_patterns = [
        r"\bby [a-z][a-z .'-]{2,40}",
        r"\breported by\b",
        r"\bwritten by\b",
        r"\bauthor:",
    ]

    if any(re.search(pattern, content + " " + title, re.IGNORECASE)
           for pattern in author_patterns):
        score += 50
        indicators.append("Possible author attribution found")

    return min(score, 100), indicators


def calculate_independence_score(domain, publisher_counts, total_publishers):
    """
    Estimate publisher independence using publisher repetition.

    This checks publisher diversity, not ownership relationships
    or whether publishers share the same original source.
    """

    publisher = get_root_publisher(domain)
    count = publisher_counts.get(publisher, 0)

    if count <= 1:
        return 100

    if total_publishers <= 1:
        return 0

    return max(0, round(100 / count, 2))


# ---------------------------------------------------------
# 4. FINAL SCORE
# ---------------------------------------------------------

def calculate_final_score(factor_scores):
    """
    Calculate a weighted score using available factors only.
    Unknown publisher reputation is excluded and remaining
    factor weights are proportionally adjusted.
    """

    available_weight = 0
    weighted_total = 0

    for factor, weight in FACTOR_WEIGHTS.items():
        score = factor_scores.get(factor)

        if score is not None:
            weighted_total += score * weight
            available_weight += weight

    if available_weight == 0:
        return None

    return round(weighted_total / available_weight, 2)


def get_score_label(score):
    if score is None:
        return "Insufficient information"

    if score >= 80:
        return "Higher indicator score"
    elif score >= 60:
        return "Moderate indicator score"
    elif score >= 40:
        return "Limited indicator score"

    return "Low indicator score"


# ---------------------------------------------------------
# 5. MAIN SOURCE ANALYSIS
# ---------------------------------------------------------

def analyze_sources(search_results):
    """
    Analyze Tavily search results.

    Expected input:
    [
        {
            "title": "...",
            "url": "...",
            "content": "...",
            "published_date": "..."
        }
    ]
    """

    if not isinstance(search_results, list):
        raise TypeError("search_results must be a list.")

    # Remove duplicate URLs
    unique_results = []
    seen_urls = set()

    for result in search_results:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url", "") or "").strip()

        if not url:
            continue

        normalized_url = url.rstrip("/").lower()

        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            unique_results.append(result)

    if not unique_results:
        return {
            "total_sources": 0,
            "unique_publishers": 0,
            "average_credibility": None,
            "diversity_status": "No sources available",
            "publisher_distribution": {},
            "repeated_publishers": {},
            "sources": [],
            "warnings": ["No valid source URLs were found."]
        }

    domains = [
        get_root_publisher(get_domain(result.get("url")))
        for result in unique_results
    ]

    publisher_counts = Counter(domains)
    total_publishers = len(publisher_counts)

    publisher_distribution = Counter()
    source_records = []
    warnings = []

    scores = []

    for result in unique_results:
        domain = get_domain(result.get("url"))
        publisher = get_root_publisher(domain)
        category = get_source_category(domain)

        reputation_score = get_credibility(domain)

        transparency_score, transparency_indicators = (
            assess_source_transparency(result)
        )

        evidence_score, evidence_indicators = (
            assess_evidence_quality(result)
        )

        publication_score, publication_indicators = (
            assess_publication_transparency(result)
        )

        independence_score = calculate_independence_score(
            domain,
            publisher_counts,
            total_publishers
        )

        factor_scores = {
            "publisher_reputation": reputation_score,
            "source_transparency": transparency_score,
            "evidence_quality": evidence_score,
            "publication_transparency": publication_score,
            "source_independence": independence_score,
        }

        final_score = calculate_final_score(factor_scores)

        if final_score is not None:
            scores.append(final_score)

        publisher_distribution[category] += 1

        source_records.append({
            "title": result.get("title", "Untitled"),
            "url": result.get("url"),
            "domain": domain,
            "publisher": publisher,
            "credibility_score": final_score,
            "credibility_label": get_score_label(final_score),
            "credibility_type": (
                "Reference baseline available"
                if reputation_score is not None
                else "Publisher reputation not assessed"
            ),
            "factor_scores": factor_scores,
            "factor_weights": FACTOR_WEIGHTS.copy(),
            "transparency_indicators": transparency_indicators,
            "evidence_indicators": evidence_indicators,
            "publication_indicators": publication_indicators,
            "category": category,
            "published_date": result.get("published_date"),
        })

    # Publisher diversity
    if total_publishers >= 5:
        diversity_status = "Good publisher diversity"
    elif total_publishers >= 3:
        diversity_status = "Moderate publisher diversity"
    else:
        diversity_status = "Limited publisher diversity"

    repeated_publishers = {
        publisher: count
        for publisher, count in publisher_counts.items()
        if count > 1
    }

    if total_publishers < 3:
        warnings.append(
            "Fewer than three unique publishers were found."
        )

    if repeated_publishers:
        warnings.append(
            "Some publishers appear multiple times. "
            "Multiple articles from one publisher are not "
            "independent confirmation."
        )

    unknown_publishers = [
        record["domain"]
        for record in source_records
        if record["factor_scores"]["publisher_reputation"] is None
    ]

    if unknown_publishers:
        warnings.append(
            "Publisher reputation was not assessed for some sources. "
            "Their final scores use the available factors only."
        )

    warnings.append(
        "Scores are heuristic indicators based on available search-result "
        "metadata and text. They do not prove factual accuracy, editorial "
        "quality, ownership independence, or journalistic reliability."
    )

    average_credibility = (
        round(sum(scores) / len(scores), 2)
        if scores else None
    )

    return {
        "total_sources": len(unique_results),
        "unique_publishers": total_publishers,
        "average_credibility": average_credibility,
        "diversity_status": diversity_status,
        "publisher_distribution": dict(publisher_distribution),
        "repeated_publishers": repeated_publishers,
        "sources": source_records,
        "warnings": warnings,
    }


# ---------------------------------------------------------
# 6. LOCAL TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_results = [
        {
            "title": "Example news report",
            "url": "https://www.reuters.com/example",
            "content": (
                "According to officials, the report was published "
                "after researchers reviewed the available data. "
                "The study and its findings were discussed in detail. "
                "Source: https://example.com/report"
            ),
            "published_date": "2026-11-10"
        },
        {
            "title": "Example fact check",
            "url": "https://www.snopes.com/example",
            "content": (
                "The claim was examined using available evidence. "
                "According to the report, researchers reviewed "
                "the statements and supporting references."
            ),
            "published_date": "2026-11-09"
        },
        {
            "title": "Example article",
            "url": "https://www.bbc.com/example",
            "content": (
                "According to officials, the report contains "
                "information about the latest developments."
            ),
            "published_date": "2026-11-08"
        },
        {
            "title": "Another report",
            "url": "https://www.reuters.com/another-example",
            "content": (
                "Officials said the investigation is ongoing."
            ),
            "published_date": "2026-11-10"
        }
    ]

    result = analyze_sources(sample_results)

    print(json.dumps(result, indent=4))
