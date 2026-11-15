
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def search_news(query, max_records=10, timespan="3m"):
    """Retrieve related news coverage from GDELT DOC API."""

    if not query or not query.strip():
        return []

    params = {
        "query": query.strip()[:300],
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(max_records, 50),
        "timespan": timespan,
        "sort": "datedesc"
    }

    response = requests.get(
        GDELT_URL,
        params=params,
        headers={"User-Agent": "NEWSGUARD-AI/1.0"},
        timeout=30
    )

    response.raise_for_status()

    try:
        data = response.json()
    except ValueError:
        raise RuntimeError(
            "GDELT returned an unexpected response. "
            "Try simplifying the search query."
        )

    articles = []

    for item in data.get("articles", []):
        articles.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "domain": item.get("domain", ""),
            "date": item.get("seendate", ""),
            "language": item.get("language", ""),
            "source_country": item.get("sourcecountry", "")
        })

    return articles
